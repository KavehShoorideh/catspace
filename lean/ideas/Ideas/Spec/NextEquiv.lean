import Chess.Legality
import Mathlib.Tactic.IntervalCases
import Ideas.Spec.Chess

/-!
# The spec's next state is a function, and it is the kernel's `applyMove` (ANALYST_PLAN 76.1 (b), extension (4))

`Ideas.Spec.Chess` is GENERATED from rules/chess.gdl: the `next` relations (`nextCell`, `nextControl`, `nextCastlingRight`, `nextEpSquare`, `nextHalfmove`) and
`applyMoveSpec p mv q` -- `q` is a position the `next` facts after the joint move `mv` describe, through the fluents' bindings. This file proves, BY REASONING:

* FUNCTIONALITY on every position and every move term: at most one unit per square (`nextCell_functional`), one side to move (`nextControl_functional`), one
  e.p. square (`nextEpSquare_functional`), one clock (`nextHalfmove_functional`); so two positions the next facts describe agree on everything the spec
  states (`applyMoveSpec_unique`: the board, the side to move, the castling rights, the e.p. square, the saturated clock).
* THE KERNEL'S NEXT STATE: on every position, for every pseudo-legal move `m` of the kernel, the kernel's `Chess.applyMove p m` is the position the spec's next
  facts after the move's term describe (`applyMoveSpec_applyMove`), and the move binding reads the term back to `m` (`moveOf_termOf`).

How: the generated tables are the kernel's arithmetic (`fileSucc_iff`, `rankSucc_iff`, `countSucc_iff`, ...: each by cases on the TABLE'S ROWS -- the eight
files, the eight ranks, the clock's hundred successors; the one place a case split is unavoidable, rule-level as in `Ideas.Spec.RookEquiv`, no position and no
square pair enumerated); every relation the next state reads gets its closed form (`castled_iff`, `passantTaken_iff`, `lifted_iff`, `landed_iff`, `nextCell_iff`,
...); the kernel's `applyBoard`, `rightsAfterMove`, `newEnPassant` and clock are read per move constructor. No `sorry`, no `native_decide`, no `decide` over
positions.
-/

namespace Ideas.NextEquiv

open Chess Ideas.Spec.Chess
open Ideas.Spec.Binding (Wing right clock MoveTerm Does moveOf termOf promoKind castleMove does_move does_promote right_surj right_inj)

/-! ## The tables are the kernel's arithmetic -/

theorem file_all (f : Fin 8) : file f := by
  obtain ⟨f, hf⟩ := f
  interval_cases f <;> constructor

theorem rank_all (r : Fin 8) : rank r := by
  obtain ⟨r, hr⟩ := r
  interval_cases r <;> constructor

theorem square_all (f r : Fin 8) : square f r := square.c1 (file_all f) (rank_all r)

theorem fileSucc_iff (a b : Fin 8) : fileSucc a b ↔ b.val = a.val + 1 := by
  constructor
  · intro h
    cases h <;> rfl
  · intro h
    obtain ⟨a, ha⟩ := a
    obtain ⟨b, hb⟩ := b
    simp only at h
    subst h
    interval_cases a <;> first | omega | constructor

theorem rankSucc_iff (a b : Fin 8) : rankSucc a b ↔ b.val = a.val + 1 := by
  constructor
  · intro h
    cases h <;> rfl
  · intro h
    obtain ⟨a, ha⟩ := a
    obtain ⟨b, hb⟩ := b
    simp only at h
    subst h
    interval_cases a <;> first | omega | constructor

/-- The clock's successor table is the kernel's `+ 1` below the limit (by cases on its hundred rows). -/
theorem countSucc_iff (n m : Nat) : countSucc n m ↔ n < 100 ∧ m = n + 1 := by
  constructor
  · intro h
    cases h <;> omega
  · rintro ⟨hn, rfl⟩
    interval_cases n <;> constructor

theorem clockLimit_iff (n : Nat) : clockLimit n ↔ n = 100 := by
  constructor
  · intro h
    cases h
    rfl
  · rintro rfl
    constructor

theorem opponent_iff (c o : Color) : opponent c o ↔ o = Color.opposite c := by
  constructor
  · intro h
    cases h <;> rfl
  · rintro rfl
    cases c <;> constructor

theorem homeRank_iff (c : Color) (r : Fin 8) : Ideas.Spec.Chess.homeRank c r ↔ r = Ideas.Spec.Binding.homeRank c := by
  constructor
  · intro h
    cases h <;> rfl
  · rintro rfl
    cases c <;> constructor

/-- A wing's squares: where its castling king lands, where its rook goes from and to. -/
def kingTo : Wing → Fin 8
  | .kingside => 6
  | .queenside => 2

def rookFrom : Wing → Fin 8
  | .kingside => 7
  | .queenside => 0

def rookTo : Wing → Fin 8
  | .kingside => 5
  | .queenside => 3

theorem castlingPath_iff (w : Wing) (kt rt rf : Fin 8) : castlingPath w kt rt rf ↔ kt = kingTo w ∧ rt = rookTo w ∧ rf = rookFrom w := by
  constructor
  · intro h
    cases h <;> exact ⟨rfl, rfl, rfl⟩
  · rintro ⟨rfl, rfl, rfl⟩
    cases w <;> constructor

theorem pawnStart_iff (c : Color) (r : Fin 8) : pawnStart c r ↔ r = (match c with | .white => 1 | .black => 6) := by
  constructor
  · intro h
    cases h <;> rfl
  · intro h
    cases c <;> simp only at h <;> subst h <;> constructor

theorem forward_iff (c : Color) (d : Dir) : forward c d ↔ d = (match c with | .white => Dir.north | .black => Dir.south) := by
  constructor
  · intro h
    cases h <;> rfl
  · intro h
    cases c <;> simp only at h <;> subst h <;> constructor

theorem occupied_iff (p : Position) (f r : Fin 8) : occupied p f r ↔ p.board (f, r) ≠ none := by
  constructor
  · rintro ⟨hb⟩ h
    rw [h] at hb
    cases hb
  · intro h
    cases hb : p.board (f, r) with
    | none => exact absurd hb h
    | some q => exact occupied.c1 (c := q.color) (k := q.kind) (by rw [hb])

theorem empty_iff (p : Position) (f r : Fin 8) : empty p f r ↔ p.board (f, r) = none := by
  constructor
  · rintro ⟨_, hn⟩
    by_contra h
    exact hn ((occupied_iff p f r).2 h)
  · intro h
    exact empty.c1 (square_all f r) (fun ho => (occupied_iff p f r).1 ho h)

/-- A step north / south along a file. -/
theorem step_north_iff (f1 r1 f2 r2 : Fin 8) : step Dir.north f1 r1 f2 r2 ↔ f2 = f1 ∧ r2.val = r1.val + 1 := by
  constructor
  · intro h
    cases h with
    | c1 _ h2 => exact ⟨rfl, (rankSucc_iff _ _).1 h2⟩
  · rintro ⟨rfl, h⟩
    exact step.c1 (file_all _) ((rankSucc_iff _ _).2 h)

theorem step_south_iff (f1 r1 f2 r2 : Fin 8) : step Dir.south f1 r1 f2 r2 ↔ f2 = f1 ∧ r1.val = r2.val + 1 := by
  constructor
  · intro h
    cases h with
    | c2 _ h2 => exact ⟨rfl, (rankSucc_iff _ _).1 h2⟩
  · rintro ⟨rfl, h⟩
    exact step.c2 (file_all _) ((rankSucc_iff _ _).2 h)

/-- The pawn taken en passant stands behind the passed-over square, from the capturer's side. -/
theorem epVictim_iff (c : Color) (f2 r2 f3 r3 : Fin 8) :
    epVictim c f2 r2 f3 r3 ↔ f3 = f2 ∧ (match c with | .white => r2.val = r3.val + 1 | .black => r3.val = r2.val + 1) := by
  constructor
  · intro h
    cases h with
    | c1 _ h2 => exact ⟨rfl, (rankSucc_iff _ _).1 h2⟩
    | c2 _ h2 => exact ⟨rfl, (rankSucc_iff _ _).1 h2⟩
  · rintro ⟨rfl, h⟩
    cases c
    · exact epVictim.c1 (file_all _) ((rankSucc_iff _ _).2 h)
    · exact epVictim.c2 (file_all _) ((rankSucc_iff _ _).2 h)

theorem rightSquare_iff (c : Color) (w : Wing) (f r : Fin 8) :
    rightSquare c w f r ↔ (f, r) = rightKingSquare (right c w) ∨ (f, r) = rightRookSquare (right c w) := by
  constructor
  · intro h
    cases h <;> simp [right, rightKingSquare, rightRookSquare]
  · intro h
    cases c <;> cases w <;>
      simp only [right, rightKingSquare, rightRookSquare, Prod.mk.injEq] at h <;>
      rcases h with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> constructor

/-! ## The relations the next state reads, in closed form (on every position, for every move term) -/

/-- The side's home rank, as the binding names it. -/
abbrev home (c : Color) : Fin 8 := Ideas.Spec.Binding.homeRank c

theorem castled_iff (p : Position) (mv : MoveTerm) (c : Color) (w : Wing) :
    castled p mv c w ↔ c = p.turn ∧ mv = .move (4, home c) (kingTo w, home c) ∧ p.board (4, home c) = some ⟨c, .king⟩ := by
  constructor
  · rintro ⟨hd, hh, hb, hp⟩
    rw [does_move] at hd
    rw [homeRank_iff] at hh
    rw [castlingPath_iff] at hp
    obtain ⟨rfl, rfl, rfl⟩ := hp
    subst hh
    exact ⟨hd.1, hd.2, hb⟩
  · rintro ⟨hc, hm, hb⟩
    exact castled.c1 (does_move.2 ⟨hc, hm⟩) ((homeRank_iff _ _).2 rfl) hb ((castlingPath_iff w _ _ _).2 ⟨rfl, rfl, rfl⟩)

theorem passantTaken_iff (p : Position) (mv : MoveTerm) (f r : Fin 8) :
    passantTaken p mv f r ↔ ∃ s t : Square, mv = .move s t ∧ p.board s = some ⟨p.turn, .pawn⟩ ∧ p.enPassant = some t ∧ p.board t = none ∧
      s.1 ≠ t.1 ∧ epVictim p.turn t.1 t.2 f r := by
  constructor
  · rintro ⟨hd, hb, he, hempty, hne, hv⟩
    rw [does_move] at hd
    obtain ⟨rfl, hm⟩ := hd
    exact ⟨_, _, hm, hb, he, (empty_iff _ _ _).1 hempty, hne, hv⟩
  · rintro ⟨⟨f1, r1⟩, ⟨f2, r2⟩, hm, hb, he, ht, hne, hv⟩
    exact passantTaken.c1 (does_move.2 ⟨rfl, hm⟩) hb he ((empty_iff _ _ _).2 ht) hne hv

theorem lifted_iff (p : Position) (mv : MoveTerm) (f r : Fin 8) :
    lifted p mv f r ↔ (∃ t : Square, mv = .move (f, r) t) ∨ (∃ (t : Square) (k : PieceKind), mv = .promote (f, r) t k) ∨
      (∃ c w, castled p mv c w ∧ f = rookFrom w ∧ r = home c) ∨ passantTaken p mv f r := by
  constructor
  · intro h
    cases h with
    | c1 hd => exact Or.inl ⟨_, (does_move.1 hd).2⟩
    | c2 hd => exact Or.inr (Or.inl ⟨_, _, (does_promote.1 hd).2⟩)
    | c3 hc hh hp =>
        rw [homeRank_iff] at hh
        rw [castlingPath_iff] at hp
        exact Or.inr (Or.inr (Or.inl ⟨_, _, hc, hp.2.2, hh⟩))
    | c4 hpt => exact Or.inr (Or.inr (Or.inr hpt))
  · rintro (⟨⟨f2, r2⟩, hm⟩ | ⟨⟨f2, r2⟩, k, hm⟩ | ⟨c, w, hc, rfl, rfl⟩ | hpt)
    · exact lifted.c1 (does_move.2 ⟨rfl, hm⟩)
    · exact lifted.c2 (does_promote.2 ⟨rfl, hm⟩)
    · exact lifted.c3 hc ((homeRank_iff _ _).2 rfl) ((castlingPath_iff w _ _ _).2 ⟨rfl, rfl, rfl⟩)
    · exact lifted.c4 hpt

theorem landed_iff (p : Position) (mv : MoveTerm) (f r : Fin 8) :
    landed p mv f r ↔ (∃ s : Square, mv = .move s (f, r)) ∨ (∃ (s : Square) (k : PieceKind), mv = .promote s (f, r) k) ∨
      (∃ c w, castled p mv c w ∧ f = rookTo w ∧ r = home c) := by
  constructor
  · intro h
    cases h with
    | c1 hd => exact Or.inl ⟨_, (does_move.1 hd).2⟩
    | c2 hd => exact Or.inr (Or.inl ⟨_, _, (does_promote.1 hd).2⟩)
    | c3 hc hh hp =>
        rw [homeRank_iff] at hh
        rw [castlingPath_iff] at hp
        exact Or.inr (Or.inr ⟨_, _, hc, hp.2.1, hh⟩)
  · rintro (⟨⟨f1, r1⟩, hm⟩ | ⟨⟨f1, r1⟩, k, hm⟩ | ⟨c, w, hc, rfl, rfl⟩)
    · exact landed.c1 (does_move.2 ⟨rfl, hm⟩)
    · exact landed.c2 (does_promote.2 ⟨rfl, hm⟩)
    · exact landed.c3 hc ((homeRank_iff _ _).2 rfl) ((castlingPath_iff w _ _ _).2 ⟨rfl, rfl, rfl⟩)

theorem touched_iff (p : Position) (mv : MoveTerm) (f r : Fin 8) : touched p mv f r ↔ lifted p mv f r ∨ landed p mv f r := by
  constructor
  · intro h
    cases h with
    | c1 h => exact Or.inl h
    | c2 h => exact Or.inr h
  · rintro (h | h)
    · exact touched.c1 h
    · exact touched.c2 h

theorem nextCell_iff (p : Position) (mv : MoveTerm) (f r : Fin 8) (c : Color) (k : PieceKind) :
    nextCell p mv f r c k ↔ (∃ s : Square, mv = .move s (f, r) ∧ c = p.turn ∧ p.board s = some ⟨c, k⟩) ∨ (∃ s : Square, mv = .promote s (f, r) k ∧ c = p.turn) ∨
      (∃ w, castled p mv c w ∧ f = rookTo w ∧ r = home c ∧ k = .rook) ∨ (p.board (f, r) = some ⟨c, k⟩ ∧ ¬ lifted p mv f r ∧ ¬ landed p mv f r) := by
  constructor
  · intro h
    cases h with
    | c1 hd hb => exact Or.inl ⟨_, (does_move.1 hd).2, (does_move.1 hd).1, hb⟩
    | c2 hd => exact Or.inr (Or.inl ⟨_, (does_promote.1 hd).2, (does_promote.1 hd).1⟩)
    | c3 hc hh hp =>
        rw [homeRank_iff] at hh
        rw [castlingPath_iff] at hp
        exact Or.inr (Or.inr (Or.inl ⟨_, hc, hp.2.1, hh, rfl⟩))
    | c4 hb hl hd => exact Or.inr (Or.inr (Or.inr ⟨hb, hl, hd⟩))
  · rintro (⟨⟨f1, r1⟩, hm, hc, hb⟩ | ⟨⟨f1, r1⟩, hm, hc⟩ | ⟨w, hc, rfl, rfl, rfl⟩ | ⟨hb, hl, hd⟩)
    · exact nextCell.c1 (does_move.2 ⟨hc, hm⟩) hb
    · exact nextCell.c2 (does_promote.2 ⟨hc, hm⟩)
    · exact nextCell.c3 hc ((homeRank_iff _ _).2 rfl) ((castlingPath_iff w _ _ _).2 ⟨rfl, rfl, rfl⟩)
    · exact nextCell.c4 hb hl hd

theorem nextControl_iff (p : Position) (c : Color) : nextControl p c ↔ c = Color.opposite p.turn := by
  constructor
  · rintro ⟨rfl, ho⟩
    exact (opponent_iff _ _).1 ho
  · rintro rfl
    exact nextControl.c1 rfl ((opponent_iff _ _).2 rfl)

theorem nextCastlingRight_iff (p : Position) (mv : MoveTerm) (c : Color) (w : Wing) :
    nextCastlingRight p mv c w ↔ right c w ∈ p.castling ∧ ¬ rightLost p mv c w := by
  constructor
  · rintro ⟨h1, h2⟩
    exact ⟨h1, h2⟩
  · rintro ⟨h1, h2⟩
    exact nextCastlingRight.c1 h1 h2

theorem rightLost_iff (p : Position) (mv : MoveTerm) (c : Color) (w : Wing) :
    rightLost p mv c w ↔ ∃ x : Square, (x = rightKingSquare (right c w) ∨ x = rightRookSquare (right c w)) ∧ touched p mv x.1 x.2 := by
  constructor
  · rintro ⟨hs, ht⟩
    exact ⟨_, (rightSquare_iff _ _ _ _).1 hs, ht⟩
  · rintro ⟨⟨f, r⟩, hs, ht⟩
    exact rightLost.c1 ((rightSquare_iff _ _ _ _).2 hs) ht

theorem nextEpSquare_iff (p : Position) (mv : MoveTerm) (f r : Fin 8) :
    nextEpSquare p mv f r ↔ ∃ r1 r3 : Fin 8, mv = .move (f, r1) (f, r3) ∧ p.board (f, r1) = some ⟨p.turn, .pawn⟩ ∧
      (match p.turn with | .white => r1.val = 1 ∧ r.val = 2 ∧ r3.val = 3 | .black => r1.val = 6 ∧ r.val = 5 ∧ r3.val = 4) := by
  constructor
  · rintro ⟨hd, hb, hs, hf, h1, h2⟩
    rw [does_move] at hd
    obtain ⟨rfl, hm⟩ := hd
    refine ⟨_, _, hm, hb, ?_⟩
    rw [pawnStart_iff] at hs
    rw [forward_iff] at hf
    cases hc : p.turn <;> rw [hc] at hs hf <;> simp only at hs hf <;> subst hs <;> subst hf
    · rw [step_north_iff] at h1 h2
      simp only
      omega
    · rw [step_south_iff] at h1 h2
      simp only
      omega
  · rintro ⟨r1, r3, hm, hb, hr⟩
    cases hc : p.turn <;> rw [hc] at hr hb <;> simp only at hr
    · exact nextEpSquare.c1 (d := Dir.north) (does_move.2 ⟨hc.symm, hm⟩) hb ((pawnStart_iff _ _).2 (Fin.ext (by simp; omega)))
        ((forward_iff _ _).2 rfl) ((step_north_iff _ _ _ _).2 ⟨rfl, by omega⟩) ((step_north_iff _ _ _ _).2 ⟨rfl, by omega⟩)
    · exact nextEpSquare.c1 (d := Dir.south) (does_move.2 ⟨hc.symm, hm⟩) hb ((pawnStart_iff _ _).2 (Fin.ext (by simp; omega)))
        ((forward_iff _ _).2 rfl) ((step_south_iff _ _ _ _).2 ⟨rfl, by omega⟩) ((step_south_iff _ _ _ _).2 ⟨rfl, by omega⟩)

theorem clockReset_iff (p : Position) (mv : MoveTerm) :
    clockReset p mv ↔ (∃ s t : Square, mv = .move s t ∧ (p.board s = some ⟨p.turn, .pawn⟩ ∨ p.board t ≠ none)) ∨
      (∃ (s t : Square) (k : PieceKind), mv = .promote s t k) := by
  constructor
  · rintro ⟨he⟩
    cases he with
    | c1 hd hb => obtain ⟨rfl, hm⟩ := does_move.1 hd; exact Or.inl ⟨_, _, hm, Or.inl hb⟩
    | c2 hd => exact Or.inr ⟨_, _, _, (does_promote.1 hd).2⟩
    | c3 hd ho => exact Or.inl ⟨_, _, (does_move.1 hd).2, Or.inr ((occupied_iff _ _ _).1 ho)⟩
    | c4 hd _ => exact Or.inr ⟨_, _, _, (does_promote.1 hd).2⟩
  · rintro (⟨⟨f1, r1⟩, ⟨f2, r2⟩, hm, hb | ho⟩ | ⟨⟨f1, r1⟩, ⟨f2, r2⟩, k, hm⟩)
    · exact clockReset.c1 (clockEvent.c1 (does_move.2 ⟨rfl, hm⟩) hb)
    · exact clockReset.c1 (clockEvent.c3 (does_move.2 ⟨rfl, hm⟩) ((occupied_iff _ _ _).2 ho))
    · exact clockReset.c1 (clockEvent.c2 (does_promote.2 ⟨rfl, hm⟩))

theorem clock_le (p : Position) : clock p ≤ 100 := by
  simp only [Ideas.Spec.Binding.clock]
  omega

theorem nextHalfmove_iff (p : Position) (mv : MoveTerm) (n : Nat) :
    nextHalfmove p mv n ↔ (clockReset p mv ∧ n = 0) ∨ (¬ clockReset p mv ∧ n = min (clock p + 1) 100) := by
  constructor
  · intro h
    cases h with
    | c1 hr => exact Or.inl ⟨hr, rfl⟩
    | c2 hc hs hr =>
        rw [countSucc_iff] at hs
        refine Or.inr ⟨hr, ?_⟩
        omega
    | c3 hl hc hr =>
        rw [clockLimit_iff] at hl
        refine Or.inr ⟨hr, ?_⟩
        omega
  · rintro (⟨hr, rfl⟩ | ⟨hr, rfl⟩)
    · exact nextHalfmove.c1 hr
    · have hle := clock_le p
      by_cases hlt : clock p < 100
      · have e : min (clock p + 1) 100 = clock p + 1 := by omega
        rw [e]
        exact nextHalfmove.c2 rfl ((countSucc_iff _ _).2 ⟨hlt, rfl⟩) hr
      · have e : min (clock p + 1) 100 = 100 := by omega
        have e2 : clock p = 100 := by omega
        rw [e]
        exact nextHalfmove.c3 ((clockLimit_iff _).2 rfl) e2 hr

/-! ## (4) THE NEXT STATE IS A FUNCTION (on every position, for every move term) -/

theorem kingTo_ne_rookTo (w w' : Wing) : kingTo w ≠ rookTo w' := by
  intro h
  cases w <;> cases w' <;> simp [kingTo, rookTo, Fin.ext_iff] at h

theorem move_ne_promote {s t s' t' : Square} {k : PieceKind} : (MoveTerm.move s t) ≠ MoveTerm.promote s' t' k := by
  intro h
  cases h

/-- AT MOST ONE UNIT PER SQUARE after the move. -/
theorem nextCell_functional (p : Position) (mv : MoveTerm) (f r : Fin 8) {c c' : Color} {k k' : PieceKind}
    (h : nextCell p mv f r c k) (h' : nextCell p mv f r c' k') : c = c' ∧ k = k' := by
  have land1 : ∀ s, mv = .move s (f, r) → landed p mv f r := fun s hm => (landed_iff _ _ _ _).2 (Or.inl ⟨s, hm⟩)
  have land2 : ∀ s k, mv = .promote s (f, r) k → landed p mv f r := fun s k hm => (landed_iff _ _ _ _).2 (Or.inr (Or.inl ⟨s, k, hm⟩))
  have land3 : ∀ c w, castled p mv c w → f = rookTo w → r = home c → landed p mv f r :=
    fun c w hc h1 h2 => (landed_iff _ _ _ _).2 (Or.inr (Or.inr ⟨c, w, hc, h1, h2⟩))
  -- a castling's king lands where no rook goes: the move's landing square is not the rook's
  have kr : ∀ s w c, mv = .move s (f, r) → castled p mv c w → f = rookTo w → False := by
    intro s w c hm hc hf
    obtain ⟨_, hm2, _⟩ := (castled_iff _ _ _ _).1 hc
    rw [hm] at hm2
    simp only [MoveTerm.move.injEq, Prod.mk.injEq] at hm2
    exact kingTo_ne_rookTo w w (hm2.2.1.symm.trans hf)
  rw [nextCell_iff] at h h'
  rcases h with ⟨s, hm, hc, hb⟩ | ⟨s, hm, hc⟩ | ⟨w, hcs, hf, hr, hk⟩ | ⟨hb, hl, hd⟩ <;>
    rcases h' with ⟨s', hm', hc', hb'⟩ | ⟨s', hm', hc'⟩ | ⟨w', hcs', hf', hr', hk'⟩ | ⟨hb', hl', hd'⟩
  · rw [hm] at hm'
    simp only [MoveTerm.move.injEq] at hm'
    rw [hm'.1] at hb
    rw [hb] at hb'
    simp only [Option.some.injEq, Piece.mk.injEq] at hb'
    exact hb'
  · exact absurd (hm.symm.trans hm') move_ne_promote
  · exact (kr s w' c' hm hcs' hf').elim
  · exact absurd (land1 s hm) hd'
  · exact absurd (hm'.symm.trans hm) move_ne_promote
  · rw [hm] at hm'
    simp only [MoveTerm.promote.injEq] at hm'
    exact ⟨hc.trans hc'.symm, hm'.2.2⟩
  · obtain ⟨_, hm2, _⟩ := (castled_iff _ _ _ _).1 hcs'
    exact absurd (hm2.symm.trans hm) move_ne_promote
  · exact absurd (land2 s k hm) hd'
  · exact (kr s' w c hm' hcs hf).elim
  · obtain ⟨_, hm2, _⟩ := (castled_iff _ _ _ _).1 hcs
    exact absurd (hm2.symm.trans hm') move_ne_promote
  · obtain ⟨hc1, _, _⟩ := (castled_iff _ _ _ _).1 hcs
    obtain ⟨hc2, _, _⟩ := (castled_iff _ _ _ _).1 hcs'
    exact ⟨hc1.trans hc2.symm, hk.trans hk'.symm⟩
  · exact absurd (land3 c w hcs hf hr) hd'
  · exact absurd (land1 s' hm') hd
  · exact absurd (land2 s' k' hm') hd
  · exact absurd (land3 c' w' hcs' hf' hr') hd
  · rw [hb] at hb'
    simp only [Option.some.injEq, Piece.mk.injEq] at hb'
    exact hb'

/-- ONE SIDE TO MOVE after the move. -/
theorem nextControl_functional (p : Position) {c c' : Color} (h : nextControl p c) (h' : nextControl p c') : c = c' := by
  rw [nextControl_iff] at h h'
  exact h.trans h'.symm

/-- AT MOST ONE E.P. SQUARE after the move. -/
theorem nextEpSquare_functional (p : Position) (mv : MoveTerm) {f r f' r' : Fin 8} (h : nextEpSquare p mv f r) (h' : nextEpSquare p mv f' r') :
    f = f' ∧ r = r' := by
  rw [nextEpSquare_iff] at h h'
  obtain ⟨r1, r3, hm, _, hr⟩ := h
  obtain ⟨r1', r3', hm', _, hr'⟩ := h'
  rw [hm] at hm'
  simp only [MoveTerm.move.injEq, Prod.mk.injEq] at hm'
  refine ⟨hm'.1.1, ?_⟩
  cases hc : p.turn <;> rw [hc] at hr hr' <;> simp only at hr hr' <;> exact Fin.ext (by omega)

/-- ONE CLOCK after the move. -/
theorem nextHalfmove_functional (p : Position) (mv : MoveTerm) {n n' : Nat} (h : nextHalfmove p mv n) (h' : nextHalfmove p mv n') : n = n' := by
  rw [nextHalfmove_iff] at h h'
  rcases h with ⟨hr, rfl⟩ | ⟨hr, rfl⟩ <;> rcases h' with ⟨hr', rfl⟩ | ⟨hr', rfl⟩
  · rfl
  · exact absurd hr hr'
  · exact absurd hr' hr
  · rfl

/-- THE NEXT STATE IS ONE POSITION: two positions the spec's next facts describe agree on everything the spec states (the board, the side to move, the castling
rights, the e.p. square, the clock as the spec reads it). -/
theorem applyMoveSpec_unique (p : Position) (mv : MoveTerm) {q₁ q₂ : Position} (h₁ : applyMoveSpec p mv q₁) (h₂ : applyMoveSpec p mv q₂) :
    q₁.board = q₂.board ∧ q₁.turn = q₂.turn ∧ q₁.castling = q₂.castling ∧ q₁.enPassant = q₂.enPassant ∧ clock q₁ = clock q₂ := by
  obtain ⟨b₁, t₁, c₁, e₁, k₁⟩ := h₁
  obtain ⟨b₂, t₂, c₂, e₂, k₂⟩ := h₂
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · funext x
    obtain ⟨f, r⟩ := x
    cases hq : q₁.board (f, r) with
    | some pc => exact ((b₂ f r pc.color pc.kind).1 ((b₁ f r pc.color pc.kind).2 hq)).symm
    | none =>
        cases hq2 : q₂.board (f, r) with
        | none => rfl
        | some pc =>
            have := (b₁ f r pc.color pc.kind).1 ((b₂ f r pc.color pc.kind).2 hq2)
            rw [hq] at this
            cases this
  · exact ((t₂ q₁.turn).1 ((t₁ q₁.turn).2 rfl)).symm
  · ext x
    obtain ⟨c, w, rfl⟩ := right_surj x
    rw [← c₁ c w, c₂ c w]
  · cases hq : q₁.enPassant with
    | some x => exact ((e₂ x.1 x.2).1 ((e₁ x.1 x.2).2 hq)).symm
    | none =>
        cases hq2 : q₂.enPassant with
        | none => rfl
        | some x =>
            have := (e₁ x.1 x.2).1 ((e₂ x.1 x.2).2 hq2)
            rw [hq] at this
            cases this
  · exact ((k₂ (clock q₁)).1 ((k₁ (clock q₁)).2 rfl)).symm

/-! ## THE KERNEL'S NEXT STATE: the kernel's board, rights, e.p. square and clock, read per move -/

theorem boardMove_apply (b : Board) (s t x : Square) : boardMove b s t x = if x = t then b s else if x = s then none else b x := by
  simp only [boardMove, boardUpdate, Function.update_apply]

theorem boardUpdate_apply (b : Board) (s x : Square) (v : Option Piece) : boardUpdate b s v x = if x = s then v else b x := by
  simp only [boardUpdate, Function.update_apply]

/-- The kernel's squares a move touches (the source, the destination, a castling rook's two squares) and its capture square: what ends a right. -/
def kTouched (m : Move) : Finset Square :=
  {m.source, m.destination} ∪ (match castleRookSource m, castleRookDestination m with | some rs, some rd => {rs, rd} | _, _ => ∅)

theorem mem_rightsRemoved (p : Position) (m : Move) (r : CastleRight) :
    r ∈ rightsRemovedForMove p m ↔ ∃ x : Square, (x = rightKingSquare r ∨ x = rightRookSquare r) ∧ (x ∈ kTouched m ∨ moveCaptureSquare p m = some x) := by
  unfold rightsRemovedForMove
  rw [Finset.mem_union, Finset.mem_biUnion]
  constructor
  · rintro (⟨x, hx, hr⟩ | hc)
    · simp only [rightsRemovedForSquare, Finset.mem_filter, Finset.mem_univ, true_and] at hr
      exact ⟨x, hr.imp Eq.symm Eq.symm, Or.inl hx⟩
    · cases hcs : moveCaptureSquare p m with
      | none => rw [hcs] at hc; simp at hc
      | some cs =>
          rw [hcs] at hc
          simp only [rightsRemovedForSquare, Finset.mem_filter, Finset.mem_univ, true_and] at hc
          exact ⟨cs, hc.imp Eq.symm Eq.symm, Or.inr rfl⟩
  · rintro ⟨x, hr, hx | hc⟩
    · refine Or.inl ⟨x, hx, ?_⟩
      simp only [rightsRemovedForSquare, Finset.mem_filter, Finset.mem_univ, true_and]
      exact hr.imp Eq.symm Eq.symm
    · refine Or.inr ?_
      rw [hc]
      simp only [rightsRemovedForSquare, Finset.mem_filter, Finset.mem_univ, true_and]
      exact hr.imp Eq.symm Eq.symm

/-- THE RIGHTS AFTER THE MOVE: a right is kept exactly when the spec's `right_lost` does not fire, given the touched squares agree. -/
theorem rights_equiv (p : Position) (m : Move) (htouch : ∀ x : Square, touched p (termOf m) x.1 x.2 ↔ x ∈ kTouched m ∨ moveCaptureSquare p m = some x)
    (c : Color) (w : Wing) : nextCastlingRight p (termOf m) c w ↔ right c w ∈ (applyMove p m).castling := by
  rw [nextCastlingRight_iff, rightLost_iff]
  show _ ↔ right c w ∈ rightsAfterMove p m
  rw [rightsAfterMove, Finset.mem_sdiff, mem_rightsRemoved]
  simp only [htouch]

/-- THE CLOCK AFTER THE MOVE, given the clock's events agree. -/
theorem clock_equiv (p : Position) (m : Move) (hr : clockReset p (termOf m) ↔ (moveIsPawn p m || moveIsCapture p m) = true) (n : Nat) :
    nextHalfmove p (termOf m) n ↔ clock (applyMove p m) = n := by
  rw [nextHalfmove_iff]
  simp only [Ideas.Spec.Binding.clock, applyMove_halfmove]
  by_cases h : (moveIsPawn p m || moveIsCapture p m) = true
  · rw [if_pos h]
    have := hr.2 h
    constructor
    · rintro (⟨_, rfl⟩ | ⟨hn, _⟩)
      · rfl
      · exact absurd this hn
    · intro e
      exact Or.inl ⟨this, by omega⟩
  · rw [if_neg h]
    have hn : ¬ clockReset p (termOf m) := fun x => h (hr.1 x)
    constructor
    · rintro (⟨hx, _⟩ | ⟨_, rfl⟩)
      · exact absurd hx hn
      · omega
    · intro e
      exact Or.inr ⟨hn, by omega⟩

theorem control_equiv (p : Position) (m : Move) (c : Color) : nextControl p c ↔ (applyMove p m).turn = c := by
  rw [nextControl_iff, applyMove_turn]
  exact eq_comm

/-! ### a normal move -/

/-- A pseudo-legal normal move: a unit of the side to move on `s`, its geometry to `t`. -/
theorem normal_pl {p : Position} {s t : Square} (h : pseudoLegal p (.normal s t)) :
    ∃ q, p.board s = some q ∧ q.color = p.turn ∧ normalPieceGeometryB p p.turn q s t = true := by
  simp only [pseudoLegal, pseudoLegalB, normalPseudoLegalB] at h
  cases hb : p.board s with
  | none => rw [hb] at h; cases h
  | some q =>
      rw [hb] at h
      simp only [Bool.and_eq_true, beq_iff_eq] at h
      exact ⟨q, rfl, h.1.1.1, h.2⟩

/-- A normal pseudo-legal king move is one step: never the castling's two-file move. -/
theorem normal_king_step {p : Position} {s t : Square} {q : Piece} (hq : q.kind = .king) (hg : normalPieceGeometryB p p.turn q s t = true) :
    absDiff s.1 t.1 ≤ 1 := by
  simp only [normalPieceGeometryB, hq, kingGeometryB, Bool.and_eq_true, decide_eq_true_eq] at hg
  exact hg.1.2

/-- A normal pseudo-legal pawn move onto an empty square stays on its file (its diagonal move is a capture). -/
theorem normal_pawn_file {p : Position} {s t : Square} {q : Piece} (hq : q.kind = .pawn) (hg : normalPieceGeometryB p p.turn q s t = true)
    (ht : p.board t = none) : s.1 = t.1 := by
  simp only [normalPieceGeometryB, hq, ht, Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq] at hg
  rcases hg with (⟨hm, _⟩ | ⟨_, hc⟩) | ⟨⟨hd, _⟩, _⟩
  · simp only [pawnMoveGeometryB, Bool.and_eq_true, beq_iff_eq] at hm
    exact hm.1
  · cases hc
  · simp only [pawnDoubleGeometryB, Bool.and_eq_true, beq_iff_eq] at hd
    exact hd.1

theorem absDiff_four_kingTo (w : Wing) : absDiff (4 : Fin 8) (kingTo w) = 2 := by
  cases w <;> rfl

theorem not_castled_normal {p : Position} {s t : Square} (h : pseudoLegal p (.normal s t)) (c : Color) (w : Wing) : ¬ castled p (.move s t) c w := by
  intro hc
  obtain ⟨_, hm, hb⟩ := (castled_iff _ _ _ _).1 hc
  simp only [MoveTerm.move.injEq] at hm
  obtain ⟨rfl, rfl⟩ := hm
  obtain ⟨q, hq, _, hg⟩ := normal_pl h
  rw [hb] at hq
  cases hq
  have := normal_king_step rfl hg
  rw [absDiff_four_kingTo] at this
  omega

theorem not_passant_normal {p : Position} {s t : Square} (h : pseudoLegal p (.normal s t)) (f r : Fin 8) : ¬ passantTaken p (.move s t) f r := by
  intro hp
  obtain ⟨s', t', hm, hb, _, ht, hne, _⟩ := (passantTaken_iff _ _ _ _).1 hp
  simp only [MoveTerm.move.injEq] at hm
  obtain ⟨rfl, rfl⟩ := hm
  obtain ⟨q, hq, _, hg⟩ := normal_pl h
  rw [hb] at hq
  cases hq
  exact hne (normal_pawn_file rfl hg ht)

theorem lifted_normal {p : Position} {s t : Square} (h : pseudoLegal p (.normal s t)) (f r : Fin 8) : lifted p (.move s t) f r ↔ (f, r) = s := by
  rw [lifted_iff]
  constructor
  · rintro (⟨t', hm⟩ | ⟨t', k, hm⟩ | ⟨c, w, hc, _, _⟩ | hp)
    · simp only [MoveTerm.move.injEq] at hm
      exact hm.1.symm
    · exact absurd hm move_ne_promote
    · exact absurd hc (not_castled_normal h c w)
    · exact absurd hp (not_passant_normal h f r)
  · intro hs
    exact Or.inl ⟨t, by rw [hs]⟩

theorem landed_normal {p : Position} {s t : Square} (h : pseudoLegal p (.normal s t)) (f r : Fin 8) : landed p (.move s t) f r ↔ (f, r) = t := by
  rw [landed_iff]
  constructor
  · rintro (⟨s', hm⟩ | ⟨s', k, hm⟩ | ⟨c, w, hc, _, _⟩)
    · simp only [MoveTerm.move.injEq] at hm
      exact hm.2.symm
    · exact absurd hm move_ne_promote
    · exact absurd hc (not_castled_normal h c w)
  · intro ht
    exact Or.inl ⟨s, by rw [ht]⟩

theorem board_normal {p : Position} {s t : Square} (h : pseudoLegal p (.normal s t)) (f r : Fin 8) (c : Color) (k : PieceKind) :
    nextCell p (.move s t) f r c k ↔ (applyMove p (.normal s t)).board (f, r) = some ⟨c, k⟩ := by
  obtain ⟨q, hq, hqc, _⟩ := normal_pl h
  rw [nextCell_iff, lifted_normal h, landed_normal h, applyMove_board]
  simp only [applyBoard, boardMove_apply]
  constructor
  · rintro (⟨s', hm, _, hb⟩ | ⟨s', hm, _⟩ | ⟨w, hc, _⟩ | ⟨hb, hl, hd⟩)
    · simp only [MoveTerm.move.injEq] at hm
      obtain ⟨rfl, ht⟩ := hm
      rw [if_pos ht.symm]
      exact hb
    · exact absurd hm move_ne_promote
    · exact absurd hc (not_castled_normal h c w)
    · rw [if_neg hd, if_neg hl]
      exact hb
  · intro hb
    by_cases ht : (f, r) = t
    · rw [if_pos ht] at hb
      refine Or.inl ⟨s, by rw [ht], ?_, hb⟩
      rw [hq] at hb
      simp only [Option.some.injEq] at hb
      rw [← hqc, hb]
    · rw [if_neg ht] at hb
      by_cases hs : (f, r) = s
      · rw [if_pos hs] at hb
        cases hb
      · rw [if_neg hs] at hb
        exact Or.inr (Or.inr (Or.inr ⟨hb, hs, ht⟩))

theorem touched_normal {p : Position} {s t : Square} (h : pseudoLegal p (.normal s t)) (x : Square) :
    touched p (termOf (.normal s t)) x.1 x.2 ↔ x ∈ kTouched (.normal s t) ∨ moveCaptureSquare p (.normal s t) = some x := by
  show touched p (.move s t) x.1 x.2 ↔ _
  rw [touched_iff, lifted_normal h, landed_normal h]
  simp only [kTouched, Move.source, Move.destination, castleRookSource, Finset.union_empty, Finset.mem_insert, Finset.mem_singleton,
    moveCaptureSquare]
  constructor
  · rintro (h1 | h1) <;> [exact Or.inl (Or.inl h1); exact Or.inl (Or.inr h1)]
  · rintro ((h1 | h1) | h1)
    · exact Or.inl h1
    · exact Or.inr h1
    · split at h1
      · cases h1
      · simp only [Option.some.injEq] at h1
        exact Or.inr h1.symm

theorem reset_normal {p : Position} {s t : Square} (h : pseudoLegal p (.normal s t)) :
    clockReset p (termOf (.normal s t)) ↔ (moveIsPawn p (.normal s t) || moveIsCapture p (.normal s t)) = true := by
  obtain ⟨q, hq, hqc, _⟩ := normal_pl h
  show clockReset p (.move s t) ↔ _
  rw [clockReset_iff]
  simp only [moveIsPawn, movingPiece, Move.source, hq, moveIsCapture, moveCaptureSquare, Bool.or_eq_true, decide_eq_true_eq]
  constructor
  · rintro (⟨s', t', hm, hb | ho⟩ | ⟨_, _, _, hm⟩)
    · simp only [MoveTerm.move.injEq] at hm
      obtain ⟨rfl, rfl⟩ := hm
      rw [hq] at hb
      simp only [Option.some.injEq] at hb
      exact Or.inl (by rw [hb])
    · simp only [MoveTerm.move.injEq] at hm
      obtain ⟨h1, h2⟩ := hm
      subst h1 h2
      refine Or.inr ?_
      rw [if_neg ho]
      cases hb : p.board t with
      | none => exact absurd hb ho
      | some _ => simp [hb]
    · exact absurd hm move_ne_promote
  · rintro (hp | hc)
    · refine Or.inl ⟨s, t, rfl, Or.inl ?_⟩
      rw [hq]
      cases q
      simp only at hp hqc
      rw [hp, hqc]
    · refine Or.inl ⟨s, t, rfl, Or.inr ?_⟩
      intro hn
      rw [if_pos hn] at hc
      cases hc

theorem ep_normal {p : Position} {s t : Square} (h : pseudoLegal p (.normal s t)) (f r : Fin 8) :
    nextEpSquare p (termOf (.normal s t)) f r ↔ newEnPassant p (.normal s t) = some (f, r) := by
  obtain ⟨q, hq, hqc, _⟩ := normal_pl h
  show nextEpSquare p (.move s t) f r ↔ _
  rw [nextEpSquare_iff]
  simp only [newEnPassant, pawnDoubleTarget, moveIsPawn, movingPiece, Move.source, hq, decide_eq_true_eq]
  constructor
  · rintro ⟨r1, r3, hm, hb, hr⟩
    simp only [MoveTerm.move.injEq] at hm
    obtain ⟨rfl, rfl⟩ := hm
    rw [hq] at hb
    simp only [Option.some.injEq] at hb
    subst hb
    cases hc : p.turn <;> rw [hc] at hr <;> simp only at hr
    · rw [if_pos ⟨rfl, rfl, Or.inl ⟨rfl, hr.1, hr.2.2⟩⟩, if_pos rfl]
      simp only [Option.some.injEq, Prod.mk.injEq, true_and]
      exact Fin.ext (by simp; omega)
    · rw [if_pos ⟨rfl, rfl, Or.inr ⟨rfl, hr.1, hr.2.2⟩⟩, if_neg (by simp)]
      simp only [Option.some.injEq, Prod.mk.injEq, true_and]
      exact Fin.ext (by simp; omega)
  · intro he
    split at he
    · rename_i hcond
      obtain ⟨hp, hf, hcases⟩ := hcond
      have hf' : f = s.1 := by
        split at he <;> (simp only [Option.some.injEq, Prod.mk.injEq] at he; exact he.1.symm)
      subst hf'
      refine ⟨s.2, t.2, congrArg₂ MoveTerm.move rfl (Prod.ext hf.symm rfl), ?_, ?_⟩
      · show p.board s = _
        rw [hq]
        cases q
        simp only at hp hqc
        rw [hp, hqc]
      · rcases hcases with ⟨hw, h1, h3⟩ | ⟨hb, h1, h3⟩
        · rw [if_pos hw] at he
          simp only [Option.some.injEq, Prod.mk.injEq] at he
          rw [hw]
          exact ⟨h1, by rw [← he.2], h3⟩
        · rw [if_neg (by rw [hb]; simp)] at he
          simp only [Option.some.injEq, Prod.mk.injEq] at he
          rw [hb]
          exact ⟨h1, by rw [← he.2], h3⟩
    · cases he

/-! ### a promotion -/

theorem promotion_pl {p : Position} {s t : Square} {k : PromotionKind} (h : pseudoLegal p (.promotion s t k)) :
    p.board s = some ⟨p.turn, .pawn⟩ := by
  simp only [pseudoLegal, pseudoLegalB, pseudoLegalPromotionB] at h
  cases hb : p.board s with
  | none => rw [hb] at h; cases h
  | some q =>
      rw [hb] at h
      simp only [Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨⟨hc, hk⟩, _⟩, _⟩ := h
      cases q
      simp only at hc hk
      rw [hc, hk]

theorem not_castled_promote (p : Position) (s t : Square) (K : PieceKind) (c : Color) (w : Wing) : ¬ castled p (.promote s t K) c w := by
  intro hc
  obtain ⟨_, hm, _⟩ := (castled_iff _ _ _ _).1 hc
  cases hm

theorem not_passant_promote (p : Position) (s t : Square) (K : PieceKind) (f r : Fin 8) : ¬ passantTaken p (.promote s t K) f r := by
  intro hp
  obtain ⟨_, _, hm, _⟩ := (passantTaken_iff _ _ _ _).1 hp
  cases hm

theorem lifted_promote (p : Position) (s t : Square) (K : PieceKind) (f r : Fin 8) : lifted p (.promote s t K) f r ↔ (f, r) = s := by
  rw [lifted_iff]
  constructor
  · rintro (⟨t', hm⟩ | ⟨t', k, hm⟩ | ⟨c, w, hc, _, _⟩ | hp)
    · cases hm
    · simp only [MoveTerm.promote.injEq] at hm
      exact hm.1.symm
    · exact absurd hc (not_castled_promote p s t K c w)
    · exact absurd hp (not_passant_promote p s t K f r)
  · intro hs
    exact Or.inr (Or.inl ⟨t, K, by rw [hs]⟩)

theorem landed_promote (p : Position) (s t : Square) (K : PieceKind) (f r : Fin 8) : landed p (.promote s t K) f r ↔ (f, r) = t := by
  rw [landed_iff]
  constructor
  · rintro (⟨s', hm⟩ | ⟨s', k, hm⟩ | ⟨c, w, hc, _, _⟩)
    · cases hm
    · simp only [MoveTerm.promote.injEq] at hm
      exact hm.2.1.symm
    · exact absurd hc (not_castled_promote p s t K c w)
  · intro ht
    exact Or.inr (Or.inl ⟨s, K, by rw [ht]⟩)

theorem board_promotion {p : Position} {s t : Square} {pk : PromotionKind} (f r : Fin 8) (c : Color) (k : PieceKind) :
    nextCell p (termOf (.promotion s t pk)) f r c k ↔ (applyMove p (.promotion s t pk)).board (f, r) = some ⟨c, k⟩ := by
  show nextCell p (.promote s t pk.toPieceKind) f r c k ↔ _
  rw [nextCell_iff, lifted_promote, landed_promote, applyMove_board]
  simp only [applyBoard, boardUpdate_apply]
  constructor
  · rintro (⟨s', hm, _⟩ | ⟨s', hm, hc⟩ | ⟨w, hc, _⟩ | ⟨hb, hl, hd⟩)
    · cases hm
    · simp only [MoveTerm.promote.injEq] at hm
      obtain ⟨_, ht, hk⟩ := hm
      rw [if_pos ht.symm, hk, hc]
    · exact absurd hc (not_castled_promote p s t _ c w)
    · rw [if_neg hd, if_neg hl]
      exact hb
  · intro hb
    by_cases ht : (f, r) = t
    · rw [if_pos ht] at hb
      simp only [Option.some.injEq, Piece.mk.injEq] at hb
      exact Or.inr (Or.inl ⟨s, by rw [ht, hb.2], hb.1.symm⟩)
    · rw [if_neg ht] at hb
      by_cases hs : (f, r) = s
      · rw [if_pos hs] at hb
        cases hb
      · rw [if_neg hs] at hb
        exact Or.inr (Or.inr (Or.inr ⟨hb, hs, ht⟩))

theorem touched_promotion (p : Position) (s t : Square) (pk : PromotionKind) (x : Square) :
    touched p (termOf (.promotion s t pk)) x.1 x.2 ↔ x ∈ kTouched (.promotion s t pk) ∨ moveCaptureSquare p (.promotion s t pk) = some x := by
  show touched p (.promote s t pk.toPieceKind) x.1 x.2 ↔ _
  rw [touched_iff, lifted_promote, landed_promote]
  simp only [kTouched, Move.source, Move.destination, castleRookSource, Finset.union_empty, Finset.mem_insert, Finset.mem_singleton,
    moveCaptureSquare]
  constructor
  · rintro (h1 | h1) <;> [exact Or.inl (Or.inl h1); exact Or.inl (Or.inr h1)]
  · rintro ((h1 | h1) | h1)
    · exact Or.inl h1
    · exact Or.inr h1
    · split at h1
      · cases h1
      · simp only [Option.some.injEq] at h1
        exact Or.inr h1.symm

theorem reset_promotion (p : Position) (s t : Square) (pk : PromotionKind) :
    clockReset p (termOf (.promotion s t pk)) ↔ (moveIsPawn p (.promotion s t pk) || moveIsCapture p (.promotion s t pk)) = true := by
  show clockReset p (.promote s t pk.toPieceKind) ↔ _
  rw [clockReset_iff]
  simp only [moveIsPawn, Bool.true_or, iff_true]
  exact Or.inr ⟨s, t, _, rfl⟩

theorem ep_promotion (p : Position) (s t : Square) (pk : PromotionKind) (f r : Fin 8) :
    nextEpSquare p (termOf (.promotion s t pk)) f r ↔ newEnPassant p (.promotion s t pk) = some (f, r) := by
  show nextEpSquare p (.promote s t pk.toPieceKind) f r ↔ _
  rw [nextEpSquare_iff]
  simp only [newEnPassant, pawnDoubleTarget, reduceCtorEq, iff_false]
  rintro ⟨_, _, hm, _⟩
  cases hm

/-! ### an en-passant capture -/

theorem ep_pl {p : Position} {s t : Square} (h : pseudoLegal p (.enPassant s t)) :
    p.enPassant = some t ∧ p.board s = some ⟨p.turn, .pawn⟩ ∧ pawnAttackGeometryB p.turn s t = true ∧ p.board t = none ∧
      ∃ cs, epCapturedSquare p.turn t = some cs := by
  simp only [pseudoLegal, pseudoLegalB, pseudoLegalEnPassantB] at h
  cases he : p.enPassant with
  | none => rw [he] at h; cases h
  | some ep =>
      rw [he] at h
      cases hb : p.board s with
      | none => rw [hb] at h; simp at h
      | some q =>
          rw [hb] at h
          simp only [Bool.and_eq_true, beq_iff_eq] at h
          obtain ⟨hep, ⟨⟨⟨⟨hc, hk⟩, hg⟩, ht⟩, hcs⟩⟩ := h
          refine ⟨by rw [hep], ?_, hg, ht, ?_⟩
          · cases q
            simp only at hc hk
            rw [hc, hk]
          · cases hx : epCapturedSquare p.turn t with
            | none => rw [hx] at hcs; cases hcs
            | some cs => exact ⟨cs, rfl⟩

theorem pawnAttack_file_ne {c : Color} {s t : Square} (h : pawnAttackGeometryB c s t = true) : s.1 ≠ t.1 := by
  intro e
  simp only [pawnAttackGeometryB, Bool.and_eq_true, beq_iff_eq] at h
  rw [e] at h
  simp [absDiff] at h

/-- The kernel's captured square IS the spec's `ep_victim`, on the ranks where it exists. -/
theorem epVictim_cs {c : Color} {t cs : Square} (h : epCapturedSquare c t = some cs) (f r : Fin 8) : epVictim c t.1 t.2 f r ↔ (f, r) = cs := by
  rw [epVictim_iff]
  unfold epCapturedSquare at h
  split_ifs at h with h1 h2
  · simp only [Option.some.injEq] at h
    subst h
    obtain ⟨rfl, h5⟩ := h1
    simp only [Prod.mk.injEq, Fin.ext_iff]
    omega
  · simp only [Option.some.injEq] at h
    subst h
    obtain ⟨rfl, h2⟩ := h2
    simp only [Prod.mk.injEq, Fin.ext_iff]
    omega

theorem cs_ne {c : Color} {s t cs : Square} (h : epCapturedSquare c t = some cs) (hst : s.1 ≠ t.1) : cs ≠ t ∧ cs ≠ s := by
  unfold epCapturedSquare at h
  split_ifs at h with h1 h2
  · simp only [Option.some.injEq] at h
    subst h
    refine ⟨fun e => ?_, fun e => hst (congrArg Prod.fst e).symm⟩
    have := congrArg (fun x : Square => x.2.val) e
    simp only at this
    omega
  · simp only [Option.some.injEq] at h
    subst h
    refine ⟨fun e => ?_, fun e => hst (congrArg Prod.fst e).symm⟩
    have := congrArg (fun x : Square => x.2.val) e
    simp only at this
    omega

theorem not_castled_ep {p : Position} {s t : Square} (h : pseudoLegal p (.enPassant s t)) (c : Color) (w : Wing) : ¬ castled p (.move s t) c w := by
  intro hc
  obtain ⟨_, hm, hb⟩ := (castled_iff _ _ _ _).1 hc
  simp only [MoveTerm.move.injEq] at hm
  obtain ⟨rfl, rfl⟩ := hm
  rw [(ep_pl h).2.1] at hb
  simp at hb

theorem passant_ep {p : Position} {s t cs : Square} (h : pseudoLegal p (.enPassant s t)) (hcs : epCapturedSquare p.turn t = some cs) (f r : Fin 8) :
    passantTaken p (.move s t) f r ↔ (f, r) = cs := by
  obtain ⟨he, hb, hg, ht, _⟩ := ep_pl h
  rw [passantTaken_iff]
  constructor
  · rintro ⟨s', t', hm, _, _, _, _, hv⟩
    simp only [MoveTerm.move.injEq] at hm
    obtain ⟨rfl, rfl⟩ := hm
    exact (epVictim_cs hcs f r).1 hv
  · intro hx
    exact ⟨s, t, rfl, hb, he, ht, pawnAttack_file_ne hg, (epVictim_cs hcs f r).2 hx⟩

theorem lifted_ep {p : Position} {s t cs : Square} (h : pseudoLegal p (.enPassant s t)) (hcs : epCapturedSquare p.turn t = some cs) (f r : Fin 8) :
    lifted p (.move s t) f r ↔ (f, r) = s ∨ (f, r) = cs := by
  rw [lifted_iff, passant_ep h hcs]
  constructor
  · rintro (⟨t', hm⟩ | ⟨t', k, hm⟩ | ⟨c, w, hc, _, _⟩ | hp)
    · simp only [MoveTerm.move.injEq] at hm
      exact Or.inl hm.1.symm
    · exact absurd hm move_ne_promote
    · exact absurd hc (not_castled_ep h c w)
    · exact Or.inr hp
  · rintro (hs | hx)
    · exact Or.inl ⟨t, by rw [hs]⟩
    · exact Or.inr (Or.inr (Or.inr hx))

theorem landed_ep {p : Position} {s t : Square} (h : pseudoLegal p (.enPassant s t)) (f r : Fin 8) : landed p (.move s t) f r ↔ (f, r) = t := by
  rw [landed_iff]
  constructor
  · rintro (⟨s', hm⟩ | ⟨s', k, hm⟩ | ⟨c, w, hc, _, _⟩)
    · simp only [MoveTerm.move.injEq] at hm
      exact hm.2.symm
    · exact absurd hm move_ne_promote
    · exact absurd hc (not_castled_ep h c w)
  · intro ht
    exact Or.inl ⟨s, by rw [ht]⟩

theorem board_ep {p : Position} {s t : Square} (h : pseudoLegal p (.enPassant s t)) (f r : Fin 8) (c : Color) (k : PieceKind) :
    nextCell p (termOf (.enPassant s t)) f r c k ↔ (applyMove p (.enPassant s t)).board (f, r) = some ⟨c, k⟩ := by
  obtain ⟨_, hb, hg, _, cs, hcs⟩ := ep_pl h
  obtain ⟨hct, hcs'⟩ := cs_ne hcs (pawnAttack_file_ne hg)
  show nextCell p (.move s t) f r c k ↔ _
  rw [nextCell_iff, lifted_ep h hcs, landed_ep h, applyMove_board]
  simp only [applyBoard, hcs, boardUpdate_apply, boardMove_apply]
  constructor
  · rintro (⟨s', hm, rfl, hb'⟩ | ⟨s', hm, _⟩ | ⟨w, hc, _⟩ | ⟨hb', hl, hd⟩)
    · simp only [MoveTerm.move.injEq] at hm
      obtain ⟨rfl, ht⟩ := hm
      rw [if_neg (fun e => hct (e.symm.trans ht.symm)), if_pos ht.symm]
      exact hb'
    · exact absurd hm move_ne_promote
    · exact absurd hc (not_castled_ep h c w)
    · rw [if_neg (fun e => hl (Or.inr e)), if_neg hd, if_neg (fun e => hl (Or.inl e))]
      exact hb'
  · intro hx
    by_cases hc : (f, r) = cs
    · rw [if_pos hc] at hx
      cases hx
    · rw [if_neg hc] at hx
      by_cases ht : (f, r) = t
      · rw [if_pos ht, hb] at hx
        simp only [Option.some.injEq, Piece.mk.injEq] at hx
        obtain ⟨rfl, rfl⟩ := hx
        exact Or.inl ⟨s, by rw [ht], rfl, hb⟩
      · rw [if_neg ht] at hx
        by_cases hs : (f, r) = s
        · rw [if_pos hs] at hx
          cases hx
        · rw [if_neg hs] at hx
          exact Or.inr (Or.inr (Or.inr ⟨hx, fun e => e.elim hs hc, ht⟩))

theorem touched_ep {p : Position} {s t : Square} (h : pseudoLegal p (.enPassant s t)) (x : Square) :
    touched p (termOf (.enPassant s t)) x.1 x.2 ↔ x ∈ kTouched (.enPassant s t) ∨ moveCaptureSquare p (.enPassant s t) = some x := by
  obtain ⟨_, _, _, _, cs, hcs⟩ := ep_pl h
  show touched p (.move s t) x.1 x.2 ↔ _
  rw [touched_iff, lifted_ep h hcs, landed_ep h]
  simp only [kTouched, Move.source, Move.destination, castleRookSource, Finset.union_empty, Finset.mem_insert, Finset.mem_singleton,
    moveCaptureSquare, hcs, Option.some.injEq]
  constructor
  · rintro ((h1 | h1) | h1)
    · exact Or.inl (Or.inl h1)
    · exact Or.inr h1.symm
    · exact Or.inl (Or.inr h1)
  · rintro ((h1 | h1) | h1)
    · exact Or.inl (Or.inl h1)
    · exact Or.inr h1
    · exact Or.inl (Or.inr h1.symm)

theorem reset_ep {p : Position} {s t : Square} (h : pseudoLegal p (.enPassant s t)) :
    clockReset p (termOf (.enPassant s t)) ↔ (moveIsPawn p (.enPassant s t) || moveIsCapture p (.enPassant s t)) = true := by
  show clockReset p (.move s t) ↔ _
  rw [clockReset_iff]
  simp only [moveIsPawn, Bool.true_or, iff_true]
  exact Or.inl ⟨s, t, rfl, Or.inl (ep_pl h).2.1⟩

theorem ep_ep {p : Position} {s t : Square} (h : pseudoLegal p (.enPassant s t)) (f r : Fin 8) :
    nextEpSquare p (termOf (.enPassant s t)) f r ↔ newEnPassant p (.enPassant s t) = some (f, r) := by
  show nextEpSquare p (.move s t) f r ↔ _
  rw [nextEpSquare_iff]
  simp only [newEnPassant, pawnDoubleTarget, reduceCtorEq, iff_false]
  rintro ⟨_, _, hm, _⟩
  simp only [MoveTerm.move.injEq] at hm
  obtain ⟨rfl, rfl⟩ := hm
  exact pawnAttack_file_ne (ep_pl h).2.2.1 rfl

/-! ### a castling -/

theorem castle_source (c : Color) (w : Wing) : (castleMove c w).source = (4, home c) := by cases c <;> cases w <;> rfl
theorem castle_destination (c : Color) (w : Wing) : (castleMove c w).destination = (kingTo w, home c) := by cases c <;> cases w <;> rfl
theorem castle_rookSource (c : Color) (w : Wing) : castleRookSource (castleMove c w) = some (rookFrom w, home c) := by cases c <;> cases w <;> rfl
theorem castle_rookDestination (c : Color) (w : Wing) : castleRookDestination (castleMove c w) = some (rookTo w, home c) := by
  cases c <;> cases w <;> rfl
theorem termOf_castle (c : Color) (w : Wing) : termOf (castleMove c w) = .move (4, home c) (kingTo w, home c) := by cases c <;> cases w <;> rfl

theorem applyBoard_castle (p : Position) (c : Color) (w : Wing) :
    applyBoard p (castleMove c w) = boardMove (boardMove p.board (4, home c) (kingTo w, home c)) (rookFrom w, home c) (rookTo w, home c) := by
  cases c <;> cases w <;> rfl

/-- A pseudo-legal castling: the side to move's, its right kept, king and rook home, the king's and the rook's landing squares empty. -/
theorem castle_pl {p : Position} {c : Color} {w : Wing} (h : pseudoLegal p (castleMove c w)) :
    p.turn = c ∧ right c w ∈ p.castling ∧ p.board (4, home c) = some ⟨c, .king⟩ ∧ p.board (rookFrom w, home c) = some ⟨c, .rook⟩ ∧
      p.board (kingTo w, home c) = none ∧ p.board (rookTo w, home c) = none := by
  cases c <;> cases w <;>
    simp only [castleMove, pseudoLegal, pseudoLegalB, pseudoLegalCastleB, castleColorOf, castleRightOf, castleRookSource, castleClearB,
      castleEmptySquares, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, List.all_cons, List.all_nil, Bool.and_true, reduceIte, reduceCtorEq] at h <;>
    exact ⟨h.1.1.1.1, h.1.1.1.2, h.1.1.2, h.1.2, by simp_all [kingTo, home, Ideas.Spec.Binding.homeRank], by simp_all [rookTo, home, Ideas.Spec.Binding.homeRank]⟩

theorem castle_squares (c : Color) (w : Wing) :
    (4, home c) ≠ ((kingTo w, home c) : Square) ∧ (4, home c) ≠ ((rookFrom w, home c) : Square) ∧ (4, home c) ≠ ((rookTo w, home c) : Square) ∧
      ((kingTo w, home c) : Square) ≠ (rookFrom w, home c) ∧ ((kingTo w, home c) : Square) ≠ (rookTo w, home c) ∧
      ((rookFrom w, home c) : Square) ≠ (rookTo w, home c) := by
  cases w <;> simp [kingTo, rookFrom, rookTo, Prod.ext_iff, Fin.ext_iff]

theorem kingTo_inj {w w' : Wing} (h : kingTo w = kingTo w') : w = w' := by
  cases w <;> cases w' <;> simp_all [kingTo, Fin.ext_iff]

theorem castled_castle {p : Position} {c : Color} {w : Wing} (h : pseudoLegal p (castleMove c w)) (c' : Color) (w' : Wing) :
    castled p (.move (4, home c) (kingTo w, home c)) c' w' ↔ c' = c ∧ w' = w := by
  obtain ⟨hturn, _, hk, _⟩ := castle_pl h
  rw [castled_iff]
  constructor
  · rintro ⟨hc, hm, _⟩
    have hcc : c' = c := hc.trans hturn
    subst hcc
    simp only [MoveTerm.move.injEq, Prod.mk.injEq, true_and] at hm
    exact ⟨rfl, kingTo_inj hm.1.symm⟩
  · rintro ⟨rfl, rfl⟩
    exact ⟨hturn.symm, rfl, hk⟩

theorem not_passant_castle {p : Position} {c : Color} {w : Wing} (h : pseudoLegal p (castleMove c w)) (f r : Fin 8) :
    ¬ passantTaken p (.move (4, home c) (kingTo w, home c)) f r := by
  intro hp
  obtain ⟨_, _, hk, _⟩ := castle_pl h
  obtain ⟨s', t', hm, hb, _⟩ := (passantTaken_iff _ _ _ _).1 hp
  simp only [MoveTerm.move.injEq] at hm
  obtain ⟨rfl, rfl⟩ := hm
  rw [hk] at hb
  simp at hb

theorem lifted_castle {p : Position} {c : Color} {w : Wing} (h : pseudoLegal p (castleMove c w)) (f r : Fin 8) :
    lifted p (.move (4, home c) (kingTo w, home c)) f r ↔ (f, r) = (4, home c) ∨ (f, r) = (rookFrom w, home c) := by
  rw [lifted_iff]
  constructor
  · rintro (⟨t', hm⟩ | ⟨t', k, hm⟩ | ⟨c', w', hc, hf, hr⟩ | hp)
    · simp only [MoveTerm.move.injEq] at hm
      exact Or.inl hm.1.symm
    · exact absurd hm move_ne_promote
    · obtain ⟨rfl, rfl⟩ := (castled_castle h c' w').1 hc
      exact Or.inr (by rw [hf, hr])
    · exact absurd hp (not_passant_castle h f r)
  · rintro (hs | hx)
    · exact Or.inl ⟨_, by rw [hs]⟩
    · simp only [Prod.mk.injEq] at hx
      exact Or.inr (Or.inr (Or.inl ⟨c, w, (castled_castle h c w).2 ⟨rfl, rfl⟩, hx.1, hx.2⟩))

theorem landed_castle {p : Position} {c : Color} {w : Wing} (h : pseudoLegal p (castleMove c w)) (f r : Fin 8) :
    landed p (.move (4, home c) (kingTo w, home c)) f r ↔ (f, r) = (kingTo w, home c) ∨ (f, r) = (rookTo w, home c) := by
  rw [landed_iff]
  constructor
  · rintro (⟨s', hm⟩ | ⟨s', k, hm⟩ | ⟨c', w', hc, hf, hr⟩)
    · simp only [MoveTerm.move.injEq] at hm
      exact Or.inl hm.2.symm
    · exact absurd hm move_ne_promote
    · obtain ⟨rfl, rfl⟩ := (castled_castle h c' w').1 hc
      exact Or.inr (by rw [hf, hr])
  · rintro (ht | hx)
    · exact Or.inl ⟨_, by rw [ht]⟩
    · simp only [Prod.mk.injEq] at hx
      exact Or.inr (Or.inr ⟨c, w, (castled_castle h c w).2 ⟨rfl, rfl⟩, hx.1, hx.2⟩)

theorem board_castle {p : Position} {c : Color} {w : Wing} (h : pseudoLegal p (castleMove c w)) (f r : Fin 8) (c' : Color) (k : PieceKind) :
    nextCell p (termOf (castleMove c w)) f r c' k ↔ (applyMove p (castleMove c w)).board (f, r) = some ⟨c', k⟩ := by
  obtain ⟨hturn, _, hk, hr, _, _⟩ := castle_pl h
  obtain ⟨d1, d2, d3, d4, d5, d6⟩ := castle_squares c w
  rw [termOf_castle, nextCell_iff, lifted_castle h, landed_castle h, applyMove_board, applyBoard_castle]
  simp only [boardMove_apply, if_neg d4.symm, if_neg d2.symm, hr]
  constructor
  · rintro (⟨s', hm, hc, hb⟩ | ⟨s', hm, _⟩ | ⟨w', hc, hf, hr', hkk⟩ | ⟨hb, hl, hd⟩)
    · simp only [MoveTerm.move.injEq] at hm
      obtain ⟨rfl, ht⟩ := hm
      rw [← ht, if_neg d5, if_neg d4, if_pos rfl]
      exact hb
    · exact absurd hm move_ne_promote
    · obtain ⟨rfl, rfl⟩ := (castled_castle h c' w').1 hc
      rw [hf, hr', hkk, if_pos rfl]
    · push Not at hl hd
      rw [if_neg hd.2, if_neg hl.2, if_neg hd.1, if_neg hl.1]
      exact hb
  · intro hx
    by_cases e1 : (f, r) = (rookTo w, home c)
    · rw [if_pos e1] at hx
      simp only [Option.some.injEq, Piece.mk.injEq] at hx
      obtain ⟨rfl, rfl⟩ := hx
      simp only [Prod.mk.injEq] at e1
      exact Or.inr (Or.inr (Or.inl ⟨w, (castled_castle h c w).2 ⟨rfl, rfl⟩, e1.1, e1.2, rfl⟩))
    · rw [if_neg e1] at hx
      by_cases e2 : (f, r) = (rookFrom w, home c)
      · rw [if_pos e2] at hx
        cases hx
      · rw [if_neg e2] at hx
        by_cases e3 : (f, r) = (kingTo w, home c)
        · rw [if_pos e3, hk] at hx
          simp only [Option.some.injEq, Piece.mk.injEq] at hx
          obtain ⟨rfl, rfl⟩ := hx
          exact Or.inl ⟨(4, home c), by rw [e3], hturn.symm, hk⟩
        · rw [if_neg e3] at hx
          by_cases e4 : (f, r) = (4, home c)
          · rw [if_pos e4] at hx
            cases hx
          · rw [if_neg e4] at hx
            exact Or.inr (Or.inr (Or.inr ⟨hx, fun e => e.elim e4 e2, fun e => e.elim e3 e1⟩))

theorem touched_castle {p : Position} {c : Color} {w : Wing} (h : pseudoLegal p (castleMove c w)) (x : Square) :
    touched p (termOf (castleMove c w)) x.1 x.2 ↔ x ∈ kTouched (castleMove c w) ∨ moveCaptureSquare p (castleMove c w) = some x := by
  have hcap : moveCaptureSquare p (castleMove c w) = none := by cases c <;> cases w <;> rfl
  rw [termOf_castle, touched_iff, lifted_castle h, landed_castle h, hcap]
  simp only [kTouched, castle_source, castle_destination, castle_rookSource, castle_rookDestination, Finset.mem_union, Finset.mem_insert,
    Finset.mem_singleton, reduceCtorEq, or_false]
  tauto

theorem reset_castle {p : Position} {c : Color} {w : Wing} (h : pseudoLegal p (castleMove c w)) :
    clockReset p (termOf (castleMove c w)) ↔ (moveIsPawn p (castleMove c w) || moveIsCapture p (castleMove c w)) = true := by
  obtain ⟨hturn, _, hk, _, hkt, _⟩ := castle_pl h
  have hpawn : moveIsPawn p (castleMove c w) = false := by
    have e : movingPiece p (castleMove c w) = some ⟨c, .king⟩ := by rw [movingPiece, castle_source, hk]
    cases c <;> cases w <;> simp only [castleMove, moveIsPawn] at e ⊢ <;> rw [e] <;> rfl
  have hcap : moveIsCapture p (castleMove c w) = false := by cases c <;> cases w <;> rfl
  rw [hpawn, hcap, termOf_castle, clockReset_iff]
  simp only [Bool.or_false, Bool.false_eq_true, iff_false]
  rintro (⟨s', t', hm, hb | ho⟩ | ⟨_, _, _, hm⟩)
  · simp only [MoveTerm.move.injEq] at hm
    obtain ⟨rfl, rfl⟩ := hm
    rw [hk] at hb
    simp at hb
  · simp only [MoveTerm.move.injEq] at hm
    obtain ⟨rfl, rfl⟩ := hm
    exact ho hkt
  · exact move_ne_promote hm

theorem ep_castle {p : Position} {c : Color} {w : Wing} (f r : Fin 8) :
    nextEpSquare p (termOf (castleMove c w)) f r ↔ newEnPassant p (castleMove c w) = some (f, r) := by
  have hn : newEnPassant p (castleMove c w) = none := by cases c <;> cases w <;> rfl
  rw [hn, termOf_castle, nextEpSquare_iff]
  simp only [reduceCtorEq, iff_false]
  rintro ⟨_, _, hm, _⟩
  simp only [MoveTerm.move.injEq, Prod.mk.injEq] at hm
  obtain ⟨⟨h1, _⟩, ⟨h2, _⟩⟩ := hm
  have := h1.trans h2.symm
  cases w <;> simp [kingTo, Fin.ext_iff] at this

/-! ## THE THEOREMS -/

theorem castle_case {p : Position} {c : Color} {w : Wing} (h : pseudoLegal p (castleMove c w)) :
    applyMoveSpec p (termOf (castleMove c w)) (applyMove p (castleMove c w)) :=
  ⟨board_castle h, control_equiv p _, rights_equiv p _ (touched_castle h), ep_castle, clock_equiv p _ (reset_castle h)⟩

/-- THE KERNEL'S NEXT STATE IS THE SPEC'S: on every position, for every pseudo-legal move `m` of the kernel, `Chess.applyMove p m` is the position the
spec's `next` facts after the move's term describe -- its board, its side to move, its castling rights, its e.p. square and its clock as the spec reads it. -/
theorem applyMoveSpec_applyMove (p : Position) (m : Move) (h : pseudoLegal p m) : applyMoveSpec p (termOf m) (applyMove p m) := by
  cases m with
  | normal s t => exact ⟨board_normal h, control_equiv p _, rights_equiv p _ (touched_normal h), ep_normal h, clock_equiv p _ (reset_normal h)⟩
  | promotion s t pk =>
      exact ⟨board_promotion, control_equiv p _, rights_equiv p _ (touched_promotion p s t pk), ep_promotion p s t pk, clock_equiv p _ (reset_promotion p s t pk)⟩
  | enPassant s t => exact ⟨board_ep h, control_equiv p _, rights_equiv p _ (touched_ep h), ep_ep h, clock_equiv p _ (reset_ep h)⟩
  | whiteKingCastle => exact castle_case (c := .white) (w := .kingside) h
  | whiteQueenCastle => exact castle_case (c := .white) (w := .queenside) h
  | blackKingCastle => exact castle_case (c := .black) (w := .kingside) h
  | blackQueenCastle => exact castle_case (c := .black) (w := .queenside) h

/-- THE NEXT STATE IS A FUNCTION AND IT IS `applyMove`: every position the spec's next facts describe after a pseudo-legal move's term agrees with the
kernel's `applyMove` on everything the spec states. -/
theorem applyMoveSpec_eq_applyMove (p : Position) (m : Move) (h : pseudoLegal p m) {q : Position} (hq : applyMoveSpec p (termOf m) q) :
    q.board = (applyMove p m).board ∧ q.turn = (applyMove p m).turn ∧ q.castling = (applyMove p m).castling ∧
      q.enPassant = (applyMove p m).enPassant ∧ clock q = clock (applyMove p m) :=
  applyMoveSpec_unique p (termOf m) hq (applyMoveSpec_applyMove p m h)

theorem moveOf_castle {p : Position} {c : Color} {w : Wing} (h : pseudoLegal p (castleMove c w)) : moveOf p (termOf (castleMove c w)) = some (castleMove c w) := by
  obtain ⟨hturn, _, hk, _⟩ := castle_pl h
  subst hturn
  rw [termOf_castle]
  simp only [moveOf]
  cases w <;> split_ifs <;> first | rfl | (exfalso; simp_all [kingTo])

/-- THE MOVE BINDING READS EVERY PSEUDO-LEGAL MOVE BACK: the kernel's move a term means at the position (`moveOf`, reading the position) is the move the
term was written from -- e1g1 is the castling exactly where it is pseudo-legal, the pawn's sideways move onto the empty e.p. square the en passant. -/
theorem moveOf_termOf (p : Position) (m : Move) (h : pseudoLegal p m) : moveOf p (termOf m) = some m := by
  cases m with
  | normal s t =>
      obtain ⟨q, hq, _, hg⟩ := normal_pl h
      have nk : ∀ w : Wing, ¬ (p.board s = some ⟨p.turn, .king⟩ ∧ s = (4, home p.turn) ∧ t = (kingTo w, home p.turn)) := by
        rintro w ⟨hb, rfl, rfl⟩
        rw [hq] at hb
        cases hb
        have := normal_king_step rfl hg
        rw [absDiff_four_kingTo] at this
        omega
      simp only [termOf, moveOf]
      split_ifs with h1 h2 h3
      · exact absurd h1 (nk .kingside)
      · exact absurd h2 (nk .queenside)
      · obtain ⟨hb, _, ht, hne⟩ := h3
        rw [hq] at hb
        cases hb
        exact absurd (normal_pawn_file rfl hg ht) hne
      · rfl
  | promotion s t pk => simp only [termOf, moveOf, Ideas.Spec.Binding.promoKind_toPieceKind]
  | enPassant s t =>
      obtain ⟨he, hb, hg, ht, _⟩ := ep_pl h
      simp only [termOf, moveOf]
      split_ifs with h1 h2 h3
      · rw [hb] at h1
        simp at h1
      · rw [hb] at h2
        simp at h2
      · rfl
      · exact (h3 ⟨hb, he, ht, pawnAttack_file_ne hg⟩).elim
  | whiteKingCastle => exact moveOf_castle (c := .white) (w := .kingside) h
  | whiteQueenCastle => exact moveOf_castle (c := .white) (w := .queenside) h
  | blackKingCastle => exact moveOf_castle (c := .black) (w := .kingside) h
  | blackQueenCastle => exact moveOf_castle (c := .black) (w := .queenside) h

/-! ## (5) THREEFOLD THROUGH THE GAME RECORD: the spec's `draw_by_repetition` reads the input `record` -/

/-- The spec's draw by repetition is the record's third occurrence (`Ideas.Game.Record.threefold`). -/
theorem drawByRepetition_iff (p : Position) (rec : Ideas.Game.Record Position RepetitionKey) :
    drawByRepetition p rec ↔ rec.threefold repetitionKey p = true := by
  constructor
  · rintro ⟨hl, hs⟩
    cases hl
    exact (Ideas.Spec.Binding.seen_two_iff_threefold rec p).1 hs
  · intro h
    exact drawByRepetition.c1 repetitionLimit.c1 ((Ideas.Spec.Binding.seen_two_iff_threefold rec p).2 h)

/-- With a sound record, it is the kernel's `Chess.threefoldClaimable` of the history: the positions before, then this one (`Ideas.Game.chess_threefold_iff`). -/
theorem drawByRepetition_kernel (p : Position) (rec : Ideas.Game.Record Position RepetitionKey) (h : rec.Sound repetitionKey) :
    drawByRepetition p rec ↔ threefoldClaimable (rec.before ++ [p]) := by
  rw [drawByRepetition_iff]
  exact Ideas.Game.chess_threefold_iff rec p h

end Ideas.NextEquiv
