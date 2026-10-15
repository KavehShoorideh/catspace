import Ideas.Base.Vocabulary

/-!
# THE WINDOW AS A TREE: the steps a mined premise composes into `forcingOf` (Kaveh 2026-10-15, ANALYST_PLAN 41.8)

The perturbation drill (JOURNAL 2026-10-15 14:04) showed that no premise read at ONE board is sound for the rook fork: the premise
must be read along the line -- after our move, for EVERY reply, the words at that node; at the end, their best capture. The mined
statements say that with the words `after`, `least` and `most`. These are the local theorems that turn such a statement into the
kernel's `forcingOf` (a bounded search over `forcedWithinB`): one step for our move, one for every reply of theirs, the leaf from the
material words, and the bridge from the relaxed board `after` reads to the board the kernel's `applyMove` produces.

Nothing here is a search: each theorem is one unfolding, and a proof of a mined lemma is their composition along the premise's line.
-/

namespace Ideas.TreeStep

open Chess
open Ideas.Vocabulary

/-! ## forcedWithinB, one step at a time -/

/-- The goal held now is forced within any horizon. -/
theorem leaf (c : Color) (g : Position → Bool) (p : Position) (n : Nat) (h : g p = true) :
    Ideas.forcedWithinB c g p n = true := by
  cases n with
  | zero => simpa [Ideas.forcedWithinB] using h
  | succ n => simp [Ideas.forcedWithinB, h]

/-- OUR STEP: on our turn, one legal move after which the goal is forced within `n` forces it within `n + 1`. -/
theorem step_ours (c : Color) (g : Position → Bool) (p : Position) (n : Nat) (hc : p.turn = c) (m : Move)
    (hm : m ∈ legalMoves p) (h : Ideas.forcedWithinB c g (applyMove p m) n = true) :
    Ideas.forcedWithinB c g p (n + 1) = true := by
  simp only [Ideas.forcedWithinB, hc, if_true, Bool.or_eq_true, List.any_eq_true]
  exact Or.inr ⟨m, hm, h⟩

/-- THEIR STEP: on their turn, when they have a legal move and after every one of them the goal is forced within `n`, it is forced
within `n + 1`. -/
theorem step_theirs (c : Color) (g : Position → Bool) (p : Position) (n : Nat) (hc : p.turn ≠ c)
    (hne : legalMoves p ≠ []) (h : ∀ m ∈ legalMoves p, Ideas.forcedWithinB c g (applyMove p m) n = true) :
    Ideas.forcedWithinB c g p (n + 1) = true := by
  simp only [Ideas.forcedWithinB, hc, if_false, Bool.or_eq_true, Bool.and_eq_true, Bool.not_eq_true', List.isEmpty_eq_false_iff,
    List.all_eq_true]
  exact Or.inr ⟨hne, h⟩

/-- A forced goal stays forced over a longer horizon. -/
theorem mono (c : Color) (g : Position → Bool) : ∀ (n : Nat) (p : Position),
    Ideas.forcedWithinB c g p n = true → Ideas.forcedWithinB c g p (n + 1) = true := by
  intro n
  induction n with
  | zero =>
    intro p h
    exact leaf c g p 1 (by simpa [Ideas.forcedWithinB] using h)
  | succ n ih =>
    intro p h
    simp only [Ideas.forcedWithinB, Bool.or_eq_true] at h
    rcases h with h | h
    · exact leaf c g p _ h
    · by_cases hc : p.turn = c
      · simp only [hc, if_true, List.any_eq_true] at h
        obtain ⟨m, hm, hm'⟩ := h
        exact step_ours c g p (n + 1) hc m hm (ih _ hm')
      · simp only [hc, if_false, Bool.and_eq_true, Bool.not_eq_true', List.isEmpty_eq_false_iff, List.all_eq_true] at h
        exact step_theirs c g p (n + 1) hc h.1 (fun m hm => ih _ (h.2 m hm))

/-! ## the leaf from the words -/

/-- THE LEAF: on our turn, our material over theirs at least `base + v` and the game alive is the gain goal. (Rewrites, not simp: the
other disjunct holds the move generator, which must not be evaluated.) -/
theorem gainGoal_of_material (c : Color) (base v : Int) (q : Position) (hq : q.turn = c)
    (hm : (materialOf q c : Int) - materialOf q (Color.opposite c) ≥ base + v) (hd : deadDraw q = false) :
    gainGoal c base v q = true := by
  unfold gainGoal
  have h1 : (q.turn == c) = true := by rw [beq_iff_eq]; exact hq
  have h2 : decide ((materialOf q c : Int) - materialOf q (Color.opposite c) ≥ base + v) = true := decide_eq_true hm
  rw [h1, h2, hd]
  rfl

/-- The gain goal is also met by mating them. -/
theorem gainGoal_of_mate (c : Color) (base v : Int) (q : Position) (hq : q.turn ≠ c) (hno : legalMoves q = [])
    (hchk : inCheckB q q.turn = true) : gainGoal c base v q = true := by
  unfold gainGoal
  have h1 : (q.turn == c) = false := by rw [beq_eq_false_iff_ne]; exact hq
  have h2 : (q.turn != c) = true := by rw [bne_iff_ne]; exact hq
  rw [h1, h2, hno, hchk]
  rfl

/-! ## forcingOf from the search -/

/-- On our turn `forcingOf` is the first horizon (1, 2, 3 moves) at which the search succeeds, or 0. -/
theorem forcingOf_eq (p : Position) (c : Color) (v : Int) (hc : p.turn = c) :
    forcingOf p c v = (match [1, 2, 3].find? (fun n => Ideas.forcedWithinB c
        (gainGoal c ((materialOf p c : Int) - materialOf p (Color.opposite c)) v) p (2 * n)) with
      | some n => n
      | none => 0) := by
  unfold forcingOf
  have ht : (p.turn == c) = true := by rw [beq_iff_eq]; exact hc
  rw [ht]
  rfl

/-- `forcingOf` is nonzero as soon as the search succeeds at one of its horizons, on our turn. -/
theorem forcingOf_ne_zero (p : Position) (c : Color) (v : Int) (hc : p.turn = c) (n : Nat) (hn : n ∈ [1, 2, 3])
    (h : Ideas.forcedWithinB c (gainGoal c ((materialOf p c : Int) - materialOf p (Color.opposite c)) v) p (2 * n) = true) :
    forcingOf p c v ≠ 0 := by
  rw [forcingOf_eq p c v hc]
  generalize hres : ([1, 2, 3].find? (fun k => Ideas.forcedWithinB c
      (gainGoal c ((materialOf p c : Int) - materialOf p (Color.opposite c)) v) p (2 * k))) = r
  cases r with
  | none =>
    exfalso
    have hall := List.find?_eq_none.mp hres n hn
    simp only [h, not_true_eq_false] at hall
  | some k =>
    have hk : k ∈ [1, 2, 3] := List.mem_of_find?_eq_some hres
    simp only [List.mem_cons, List.mem_nil_iff, or_false] at hk
    show k ≠ 0
    omega

/-! ## the bridge from `least` to every reply -/

/-- What `least` reads is a floor on every legal reply. -/
theorem least_floor (p : Position) (f : Position → Nat) (k : Nat) (h : leastOver p f ≥ k) (m : Move) (hm : m ∈ legalMoves p) :
    f (applyMove p m) ≥ k := le_trans h (leastOver_le p f m hm)

/-- A positive `least` means they have a legal move at all (with none it reads 0). -/
theorem least_pos_nonempty (p : Position) (f : Position → Nat) (h : leastOver p f ≥ 1) : legalMoves p ≠ [] := by
  intro hno
  simp [leastOver, hno] at h

/-! ## the bridge from `after` to the kernel's move -/

/-- When the side to move has a legal move from `x` to `y` (a promotion only to a queen), `after` reads its body on the position the
kernel's `applyMove` produces for such a move: the mined premise's words after a legal move are words about the real game. -/
theorem afterMove_legal (p : Position) (x y : Square) (f : Position → Nat) (m : Move) (hm : m ∈ legalMoves p)
    (hx : m.source = x) (hy : m.destination = y) (hk : queenOnly m = true) :
    ∃ m' ∈ legalMoves p, m'.source = x ∧ m'.destination = y ∧ afterMove p (some x) (some y) f = f (applyMove p m') := by
  have hpred : (fun m => m.source == x && m.destination == y && queenOnly m) m = true := by
    simp only [Bool.and_eq_true, beq_iff_eq]
    exact ⟨⟨hx, hy⟩, hk⟩
  have hsome : legalFromTo p x y ≠ none := by
    intro hnone
    unfold legalFromTo at hnone
    have := List.find?_eq_none.mp hnone m hm
    exact this hpred
  generalize hres : legalFromTo p x y = r at hsome
  cases r with
  | none => exact absurd rfl hsome
  | some m' =>
    have hres' : (legalMoves p).find? (fun m => m.source == x && m.destination == y && queenOnly m) = some m' := by
      unfold legalFromTo at hres; exact hres
    have hmem : m' ∈ legalMoves p := List.mem_of_find?_eq_some hres'
    have hp := List.find?_some hres'
    simp only [Bool.and_eq_true, beq_iff_eq] at hp
    refine ⟨m', hmem, hp.1.1, hp.1.2, ?_⟩
    simp only [afterMove, hres]

/-- `moves` names exactly the legal moves of the side to move, by source and destination, when the piece is its own. -/
theorem movesOf_legal (p : Position) (x y : Square) (q : Piece) (hx : p.board x = some q) (hc : q.color = p.turn)
    (h : movesOf p x y = true) : ∃ m ∈ legalMoves p, m.source = x ∧ m.destination = y := by
  unfold movesOf at h
  rw [hx] at h
  have hcb : (q.color == p.turn) = true := by rw [beq_iff_eq]; exact hc
  simp only [hcb, if_true] at h
  simp only [List.any_eq_true, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨m, hm, hs, hd⟩ := h
  exact ⟨m, hm, hs, hd⟩

end Ideas.TreeStep
