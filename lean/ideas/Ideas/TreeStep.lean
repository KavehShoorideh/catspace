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
theorem afterMove_legal (p : Position) (x y : Square) (m : Move) (hm : m ∈ legalMoves p)
    (hx : m.source = x) (hy : m.destination = y) (hk : queenOnly m = true) :
    ∃ m' ∈ legalMoves p, m'.source = x ∧ m'.destination = y ∧
      ∀ f : Position → Nat, afterMove p (some x) (some y) f = f (applyMove p m') := by
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
    refine ⟨m', hmem, hp.1.1, hp.1.2, fun f => ?_⟩
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

/-- A legal move of a piece that is not a pawn is never a promotion. -/
theorem queenOnly_of_not_pawn (p : Position) (m : Move) (hm : m ∈ legalMoves p) (q : Piece) (hx : p.board m.source = some q)
    (hq : q.kind ≠ PieceKind.pawn) : queenOnly m = true := by
  have hl : legalMove p m := by
    unfold legalMoves at hm
    have h2 := (List.mem_filter.mp hm).2
    exact @of_decide_eq_true _ (legalMoveDecidable p m) h2
  cases m with
  | promotion s t k =>
    exfalso
    have hps : pseudoLegalPromotionB p (.promotion s t k) = true := hl.1
    simp only [Move.source] at hx
    simp only [pseudoLegalPromotionB, hx, Bool.and_eq_true, beq_iff_eq] at hps
    obtain ⟨⟨⟨_, hpawn⟩, _⟩, _⟩ := hps
    exact hq hpawn
  | normal s t => rfl
  | enPassant s t => rfl
  | whiteKingCastle => rfl
  | whiteQueenCastle => rfl
  | blackKingCastle => rfl
  | blackQueenCastle => rfl

/-- THE MOVE STEP IN WORDS: `moves` says the side to move's piece (not a pawn) can go from `x` to `y`; then there is a legal move the
kernel plays, and `after` reads its body on the position the kernel produces. The composing tactic's entry for every `after` in a premise. -/
theorem after_of_moves (p : Position) (x y : Square) (q : Piece) (hx : p.board x = some q)
    (hc : q.color = p.turn) (hq : q.kind ≠ PieceKind.pawn) (h : movesOf p x y = true) :
    ∃ m ∈ legalMoves p, m.source = x ∧ m.destination = y ∧
      ∀ f : Position → Nat, afterMove p (some x) (some y) f = f (applyMove p m) := by
  obtain ⟨m, hm, hs, hd⟩ := movesOf_legal p x y q hx hc h
  have hk : queenOnly m = true := queenOnly_of_not_pawn p m hm q (by rw [hs]; exact hx) hq
  exact afterMove_legal p x y m hm hs hd hk

/-- A positive `most` has a witness: some legal move reads at least it. -/
theorem most_witness (p : Position) (f : Position → Nat) (k : Nat) (hk : k ≥ 1) (h : mostOver p f ≥ k) :
    ∃ m ∈ legalMoves p, f (applyMove p m) ≥ k := by
  unfold mostOver at h
  have key : ∀ (l : List Move) (a : Nat), (l.map (fun m => f (applyMove p m))).foldl max a ≥ k → a ≥ k ∨ ∃ m ∈ l, f (applyMove p m) ≥ k := by
    intro l
    induction l with
    | nil => intro a h; exact Or.inl h
    | cons m l ih =>
      intro a h
      simp only [List.map_cons, List.foldl_cons] at h
      rcases ih _ h with h' | ⟨m', hm', h'⟩
      · rcases le_max_iff.mp h' with h'' | h''
        · exact Or.inl h''
        · exact Or.inr ⟨m, List.mem_cons_self .., h''⟩
      · exact Or.inr ⟨m', List.mem_cons_of_mem _ hm', h'⟩
  rcases key (legalMoves p) 0 h with h0 | hw
  · omega
  · exact hw

/-- The other side's turn after our move, and ours again after its reply. -/
theorem turn_after_two (p : Position) (c : Color) (hc : p.turn = c) (m r : Move) : (applyMove (applyMove p m) r).turn = c := by
  simp only [applyMove_turn, hc]
  cases c <;> rfl

theorem turn_after_one (p : Position) (c : Color) (hc : p.turn = c) (m : Move) : (applyMove p m).turn ≠ c := by
  simp only [applyMove_turn, hc]
  cases c <;> decide

/-! ## THE VALIDATION: a two-ply tree lemma proved from the words alone

The shape every mined tree premise has, at depth one: our legal move `x -> y` by one of our pieces (not a pawn); after it, over EVERY
reply, our material at least `A`, theirs at most `B`, the game alive; and `A - B` clears the gain. Then `forcingOf` is nonzero. The
proof is the composition of the step theorems, which is what the composing tactic must do for deeper trees. -/
theorem two_ply (p : Position) (c : Color) (v : Int) (hc : p.turn = c) (x y : Square) (q : Piece) (hx : p.board x = some q)
    (hqc : q.color = c) (hq : q.kind ≠ PieceKind.pawn) (A B : Nat)
    (h1 : movesOf p x y = true)
    (h2 : afterMove p (some x) (some y) (fun p => leastOver p (fun p => materialOf p c)) ≥ A)
    (h3 : afterMove p (some x) (some y) (fun p => mostOver p (fun p => materialOf p (Color.opposite c))) ≤ B)
    (h4 : afterMove p (some x) (some y) (fun p => leastOver p (fun p => if deadDraw p then 0 else 1)) ≥ 1)
    (hAB : (A : Int) - B ≥ ((materialOf p c : Int) - materialOf p (Color.opposite c)) + v) :
    forcingOf p c v ≠ 0 := by
  obtain ⟨m, hm, -, -, hafter⟩ := after_of_moves p x y q hx (by rw [hqc, hc]) hq h1
  rw [hafter] at h2 h3 h4
  apply forcingOf_ne_zero p c v hc 1 (by simp)
  apply step_ours c _ p 1 hc m hm
  apply step_theirs c _ (applyMove p m) 0 (turn_after_one p c hc m) (least_pos_nonempty _ _ h4)
  intro r hr
  apply leaf
  apply gainGoal_of_material c _ v _ (turn_after_two p c hc m r)
  · have ha := least_floor _ _ A h2 r hr
    have hb := le_trans (le_mostOver (applyMove p m) (fun p => materialOf p (Color.opposite c)) r hr) h3
    beta_reduce at ha hb
    omega
  · have hd := least_floor _ _ 1 h4 r hr
    beta_reduce at hd
    by_contra hcon
    simp only [Bool.not_eq_false] at hcon
    simp [hcon] at hd

end Ideas.TreeStep
