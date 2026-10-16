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

/-- Every legal reply reads at least `T`, and there is one: `least` reads at least `T`. -/
theorem least_of_each (p : Position) (F : Position → Nat) (T : Nat) (hne : legalMoves p ≠ [])
    (h : ∀ m ∈ legalMoves p, F (applyMove p m) ≥ T) : leastOver p F ≥ T := by
  unfold leastOver
  have key : ∀ (l : List Move) (a : Nat), a ≥ T → (∀ m ∈ l, F (applyMove p m) ≥ T) → (l.map (fun m => F (applyMove p m))).foldl min a ≥ T := by
    intro l
    induction l with
    | nil => intro a ha _; exact ha
    | cons m l ih =>
      intro a ha hl
      simp only [List.map_cons, List.foldl_cons]
      exact ih _ (le_min ha (hl m (List.mem_cons_self ..))) (fun m' hm' => hl m' (List.mem_cons_of_mem _ hm'))
  cases hl : legalMoves p with
  | nil => exact absurd hl hne
  | cons m l =>
    simp only [List.map_cons]
    refine key l _ (h m (by rw [hl]; exact List.mem_cons_self ..)) (fun m' hm' => h m' (by rw [hl]; exact List.mem_cons_of_mem _ hm'))

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

/-! ## the words at a node, read for the proof generator

A mined premise names our move `x -> y` with three words: the piece on `x` is ours (`count (and ours (at x)) >= 1`), it is not a pawn
(`count (and ourpawns (at x)) <= 0`), and the move is legal (`count (and (moves x) (at y)) >= 1`). These lemmas read them. -/

theorem count_pos_mem (A : SquareSet) (h : count A ≥ 1) : ∃ u, A u = true := by
  unfold count at h
  obtain ⟨u, hu⟩ := List.exists_mem_of_length_pos (by omega : 0 < (allSquares.filter A).length)
  exact ⟨u, (List.mem_filter.mp hu).2⟩

theorem count_zero_not (A : SquareSet) (h : count A ≤ 0) (u : Square) : A u = false := by
  by_contra hcon
  simp only [Bool.not_eq_false] at hcon
  have hmem : u ∈ allSquares.filter A := List.mem_filter.mpr ⟨mem_allSquares u, hcon⟩
  have := List.length_pos_of_mem hmem
  unfold count at h
  omega

/-- `ours at x`: a piece of ours, not the king, stands on `x`. -/
theorem ours_at (p : Position) (c : Color) (x : Square) (h : count (fun u => piecesOf p c u && oset atSq (some x) u) ≥ 1) :
    ∃ q : Piece, p.board x = some q ∧ q.color = c ∧ q.kind ≠ PieceKind.king := by
  obtain ⟨u, hu⟩ := count_pos_mem _ h
  simp only [Bool.and_eq_true, oset, atSq, beq_iff_eq] at hu
  obtain ⟨hp, rfl⟩ := hu
  unfold piecesOf at hp
  revert hp
  cases hb : p.board u with
  | none => intro h; simp at h
  | some q =>
    intro h
    simp only [Bool.and_eq_true, decide_eq_true_eq] at h
    exact ⟨q, rfl, h.1, h.2⟩

/-- `no pawn of ours at x`: the piece there, if ours, is not a pawn. -/
theorem not_pawn_at (p : Position) (c : Color) (x : Square) (h : count (fun u => kindSet p c PieceKind.pawn u && oset atSq (some x) u) ≤ 0)
    (q : Piece) (hx : p.board x = some q) (hc : q.color = c) : q.kind ≠ PieceKind.pawn := by
  intro hk
  have := count_zero_not _ h x
  simp only [kindSet, hx, oset, atSq, beq_self_eq_true, Bool.and_true, hc, hk, Bool.and_self] at this
  exact absurd this (by simp)

/-- `x can go to y`: the `moves` word, read as a Bool. -/
theorem moves_of_count (p : Position) (x y : Square) (h : count (fun u => oset (movesOf p) (some x) u && oset atSq (some y) u) ≥ 1) :
    movesOf p x y = true := by
  obtain ⟨u, hu⟩ := count_pos_mem _ h
  simp only [Bool.and_eq_true, oset, atSq, beq_iff_eq] at hu
  obtain ⟨hm, rfl⟩ := hu
  exact hm

/-- OUR STEP IN THE WORDS: our non-pawn piece on `x` can legally go to `y`, and after the kernel's move the goal is forced within `n`;
the words after the move are read on that very position. -/
theorem step_after (c : Color) (g : Position → Bool) (p : Position) (n : Nat) (hc : p.turn = c) (x y : Square)
    (hown : count (fun u => piecesOf p c u && oset atSq (some x) u) ≥ 1)
    (hnp : count (fun u => kindSet p c PieceKind.pawn u && oset atSq (some x) u) ≤ 0)
    (hmv : count (fun u => oset (movesOf p) (some x) u && oset atSq (some y) u) ≥ 1)
    (H : ∀ m ∈ legalMoves p, (∀ f : Position → Nat, afterMove p (some x) (some y) f = f (applyMove p m)) →
      Ideas.forcedWithinB c g (applyMove p m) n = true) :
    Ideas.forcedWithinB c g p (n + 1) = true := by
  obtain ⟨q, hx, hqc, -⟩ := ours_at p c x hown
  have hq := not_pawn_at p c x hnp q hx hqc
  obtain ⟨m, hm, -, -, hafter⟩ := after_of_moves p x y q hx (by rw [hqc, hc]) hq (moves_of_count p x y hmv)
  exact step_ours c g p n hc m hm (H m hm hafter)

/-- THEIR STEP IN THE WORDS: on their turn, some `least` reads at least 1 (so they have a move), and after every reply the goal is
forced within `n`. -/
theorem step_least (c : Color) (g : Position → Bool) (p : Position) (n : Nat) (hc : p.turn ≠ c) (f : Position → Nat) (k : Nat) (hk : k ≥ 1)
    (hne : leastOver p f ≥ k) (H : ∀ r ∈ legalMoves p, Ideas.forcedWithinB c g (applyMove p r) n = true) :
    Ideas.forcedWithinB c g p (n + 1) = true :=
  step_theirs c g p n hc (least_pos_nonempty p f (le_trans hk hne)) H

/-! ## alive from material -/

/-- The squares holding a non-king piece. -/
def nonKing (q : Position) : List Square :=
  allSquares.filter (fun u => match q.board u with | some x => x.kind != PieceKind.king | none => false)

/-- What a square contributes to `materialOf q c`. -/
def contrib (q : Position) (c : Color) (u : Square) : Nat :=
  match q.board u with | some x => (if x.color == c then (pieceValue x.kind).toNat else 0) | none => 0

theorem materialOf_eq_sum (q : Position) (c : Color) : materialOf q c = (allSquares.map (contrib q c)).sum := rfl

theorem contrib_zero_of_not_nonKing (q : Position) (c : Color) (u : Square) (hu : u ∉ nonKing q) : contrib q c u = 0 := by
  unfold nonKing at hu
  unfold contrib
  cases hb : q.board u with
  | none => rfl
  | some x =>
    have : (x.kind != PieceKind.king) = false := by
      by_contra hcon
      simp only [Bool.not_eq_false] at hcon
      exact hu (List.mem_filter.mpr ⟨mem_allSquares u, by rw [hb]; exact hcon⟩)
    simp only [bne_eq_false_iff_eq] at this
    simp [this, pieceValue]

theorem contrib_le_three (q : Position) (c : Color) (u : Square) (x : Piece) (hb : q.board u = some x)
    (hk : (x.kind == PieceKind.knight || x.kind == PieceKind.bishop) = true) : contrib q c u ≤ 3 := by
  unfold contrib
  rw [hb]
  show (if x.color == c then (pieceValue x.kind).toNat else 0) ≤ 3
  simp only [Bool.or_eq_true, beq_iff_eq] at hk
  rcases hk with hk | hk <;> rw [hk] <;> simp [pieceValue] <;> try (split <;> omega)

/-- A dead draw: the non-king squares are none, or one holding a knight or a bishop. -/
theorem deadDraw_shape (q : Position) (h : deadDraw q = true) :
    nonKing q = [] ∨ ∃ u x, nonKing q = [u] ∧ q.board u = some x ∧ (x.kind == PieceKind.knight || x.kind == PieceKind.bishop) = true := by
  unfold deadDraw at h
  change (match nonKing q with
    | [] => true
    | [u] => match q.board u with | some x => x.kind == PieceKind.knight || x.kind == PieceKind.bishop | none => false
    | _ => false) = true at h
  generalize hL : nonKing q = L at h
  match L, h with
  | [], _ => exact Or.inl rfl
  | [u], h =>
    right
    refine ⟨u, ?_⟩
    have h2 : (match q.board u with
      | some x => x.kind == PieceKind.knight || x.kind == PieceKind.bishop
      | none => false) = true := h
    revert h2
    cases hb : q.board u with
    | none => intro h2; simp at h2
    | some x => intro h2; exact ⟨x, rfl, rfl, h2⟩
  | _ :: _ :: _, h => simp at h

theorem sum_le_three_of_one (q : Position) (c : Color) (u : Square) (hu : nonKing q = [u]) (h3 : contrib q c u ≤ 3) :
    ∀ (l : List Square), l.Nodup → (l.map (contrib q c)).sum ≤ 3 := by
  intro l hnd
  induction l with
  | nil => simp
  | cons v l ih =>
    simp only [List.map_cons, List.sum_cons]
    by_cases hv : v = u
    · subst hv
      have hrest : (l.map (contrib q c)).sum = 0 := by
        apply List.sum_eq_zero
        intro y hy
        obtain ⟨w, hw, rfl⟩ := List.mem_map.mp hy
        apply contrib_zero_of_not_nonKing
        rw [hu]
        simp only [List.mem_singleton]
        intro hwv
        exact (List.nodup_cons.mp hnd).1 (hwv ▸ hw)
      omega
    · have h0 : contrib q c v = 0 := by
        apply contrib_zero_of_not_nonKing
        rw [hu]; simp [hv]
      rw [h0]
      simpa using ih (List.nodup_cons.mp hnd).2

/-- In a dead draw neither side has more than a minor piece. -/
theorem material_le_of_deadDraw (q : Position) (c : Color) (h : deadDraw q = true) : materialOf q c ≤ 3 := by
  rw [materialOf_eq_sum]
  rcases deadDraw_shape q h with hnone | ⟨u, x, hu, hb, hk⟩
  · have : (allSquares.map (contrib q c)).sum = 0 := by
      apply List.sum_eq_zero
      intro y hy
      obtain ⟨w, _, rfl⟩ := List.mem_map.mp hy
      exact contrib_zero_of_not_nonKing q c w (by rw [hnone]; simp)
    omega
  · exact sum_le_three_of_one q c u hu (contrib_le_three q c u x hb hk) allSquares allSquares_nodup

theorem alive_of_material (q : Position) (c : Color) (h : materialOf q c ≥ 4) : deadDraw q = false := by
  by_contra hcon
  simp only [Bool.not_eq_false] at hcon
  have := material_le_of_deadDraw q c hcon
  omega

/-! ## the leaf from `balance`, and our step through `most` -/

/-- THE LEAF IN ONE NUMBER: on our turn, `balance` at least `64 + base + v` with `base + v ≥ 4` is the gain goal (the material clears the
gain, and that much material is no dead draw). -/
theorem gainGoal_of_balance (c : Color) (base v : Int) (q : Position) (hq : q.turn = c) (k : Nat) (hk : (k : Int) ≥ base + v) (h4 : k ≥ 4)
    (hb : balanceOf q c ≥ 64 + k) : gainGoal c base v q = true := by
  unfold balanceOf at hb
  have hm : materialOf q c ≥ materialOf q (Color.opposite c) + k := by omega
  apply gainGoal_of_material c base v q hq
  · omega
  · exact alive_of_material q c (by omega)

/-- OUR STEP THROUGH `most`: on our turn, `most F` at least `k ≥ 1` gives a move after which `F` reads at least `k`; when every such move
forces the goal within `n`, it is forced within `n + 1`. -/
theorem step_most (c : Color) (g : Position → Bool) (p : Position) (n : Nat) (hc : p.turn = c) (F : Position → Nat) (k : Nat) (hk : k ≥ 1)
    (h : mostOver p F ≥ k) (H : ∀ m ∈ legalMoves p, F (applyMove p m) ≥ k → Ideas.forcedWithinB c g (applyMove p m) n = true) :
    Ideas.forcedWithinB c g p (n + 1) = true := by
  obtain ⟨m, hm, hF⟩ := most_witness p F k hk h
  exact step_ours c g p n hc m hm (H m hm hF)

/-- THE MINIMAX TAIL, two plies: on our turn, `most (least balance)` at least `64 + base + v` (some move of ours, then whatever they
reply, the balance clears the gain) forces the goal within two plies. -/
theorem tail_two (c : Color) (base v : Int) (p : Position) (hc : p.turn = c) (k : Nat) (hk : (k : Int) ≥ base + v) (h4 : k ≥ 4)
    (h : mostOver p (fun q => leastOver q (fun r => balanceOf r c)) ≥ 64 + k) :
    Ideas.forcedWithinB c (gainGoal c base v) p 2 = true := by
  apply step_most c _ p 1 hc _ (64 + k) (by omega) h
  intro m hm hF
  have hF' : leastOver (applyMove p m) (fun r => balanceOf r c) ≥ 64 + k := hF
  apply step_least c _ (applyMove p m) 0 (turn_after_one p c hc m) _ (64 + k) (by omega) hF'
  intro r hr
  apply leaf
  have hb : balanceOf (applyMove (applyMove p m) r) c ≥ 64 + k := least_floor _ _ (64 + k) hF' r hr
  exact gainGoal_of_balance c base v _ (turn_after_two p c hc m r) k hk h4 hb

/-! ## item 0 of ANALYST_PLAN 44: a pawn may be the piece that moves, as long as it is not about to promote

`queenOnly_of_not_pawn` kept every pawn out of a named move, because `after` promotes only to a queen. A pawn that does not stand on
the rank before promotion has no promotion among its moves, so its move `x -> y` is as good as any other (a capture, a push, an en
passant: `after` plays the kernel's own move). The leg is read "no pawn of ours on `x` that stands on our seventh rank". -/

/-- A legal promotion starts from a pawn of the side to move on the rank before promotion (the seventh, from its side). -/
theorem promotion_from_seventh (p : Position) (s t : Square) (k : PromotionKind) (hm : Move.promotion s t k ∈ legalMoves p) :
    ∃ q : Piece, p.board s = some q ∧ q.color = p.turn ∧ q.kind = PieceKind.pawn ∧
      ((p.turn = Color.white ∧ s.2.val = 6) ∨ (p.turn = Color.black ∧ s.2.val = 1)) := by
  have hl : legalMove p (Move.promotion s t k) := by
    unfold legalMoves at hm
    have h2 := (List.mem_filter.mp hm).2
    exact @of_decide_eq_true _ (legalMoveDecidable p _) h2
  have hps : pseudoLegalPromotionB p (.promotion s t k) = true := hl.1
  simp only [pseudoLegalPromotionB] at hps
  cases hb : p.board s with
  | none => simp [hb] at hps
  | some q0 =>
    simp only [hb, Bool.and_eq_true, beq_iff_eq, Bool.or_eq_true] at hps
    obtain ⟨⟨⟨hcol, hpawn⟩, htr⟩, hgeo⟩ := hps
    refine ⟨q0, rfl, hcol, hpawn, ?_⟩
    cases ht : p.turn <;> rw [ht] at htr hgeo <;>
      simp only [promotionRank, Fin.ext_iff, pawnMoveGeometryB, pawnAttackGeometryB, Bool.and_eq_true, beq_iff_eq] at htr hgeo
    · left
      refine ⟨rfl, ?_⟩
      rcases hgeo with ⟨⟨_, h1⟩, _⟩ | ⟨⟨_, h1⟩, _⟩ <;> (simp at h1; omega)
    · right
      refine ⟨rfl, ?_⟩
      rcases hgeo with ⟨⟨_, h1⟩, _⟩ | ⟨⟨_, h1⟩, _⟩ <;> (simp at h1; omega)

/-- A legal move of a piece that is not about to promote is never a promotion to anything but a queen (it is no promotion at all). -/
theorem queenOnly_of_not_promoting (p : Position) (m : Move) (hm : m ∈ legalMoves p) (q : Piece) (hx : p.board m.source = some q)
    (hnp : q.kind = PieceKind.pawn → ¬ ((p.turn = Color.white ∧ m.source.2.val = 6) ∨ (p.turn = Color.black ∧ m.source.2.val = 1))) :
    queenOnly m = true := by
  cases m with
  | promotion s t k =>
    exfalso
    obtain ⟨q', hq', -, hqk, hr⟩ := promotion_from_seventh p s t k hm
    simp only [Move.source] at hx hnp
    rw [hx] at hq'
    have := Option.some.inj hq'
    subst this
    exact hnp hqk hr
  | normal s t => rfl
  | enPassant s t => rfl
  | whiteKingCastle => rfl
  | whiteQueenCastle => rfl
  | blackKingCastle => rfl
  | blackQueenCastle => rfl

/-- THE LEG IN WORDS: `no pawn of ours at x on our seventh rank` (the word `(count (and ourpawns (and (at x) (rank <seventh>))))` at most 0)
says the piece on `x`, if it is a pawn, is not about to promote. -/
theorem not_promoting_at (p : Position) (c : Color) (x : Square) (hc : p.turn = c)
    (h : count (fun u => kindSet p c PieceKind.pawn u && (oset atSq (some x) u && oset rankOf (some (named c ((0 : Fin 8), (6 : Fin 8)))) u)) ≤ 0)
    (q : Piece) (hx : p.board x = some q) (hqc : q.color = c) :
    q.kind = PieceKind.pawn → ¬ ((p.turn = Color.white ∧ x.2.val = 6) ∨ (p.turn = Color.black ∧ x.2.val = 1)) := by
  intro hk hr
  have h0 := count_zero_not _ h x
  simp only [kindSet, hx, oset, atSq, rankOf, hqc, hk, beq_self_eq_true, Bool.and_self, Bool.true_and, Bool.and_eq_false_iff,
    decide_eq_false_iff_not] at h0
  subst hc
  rcases hr with ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> rw [h1] at h0 <;> simp [named, Fin.ext_iff, Fin.rev] at h0 <;> omega

/-- `after_of_moves` for a piece that is not about to promote (a pawn may be the mover). -/
theorem after_of_moves_np (p : Position) (x y : Square) (q : Piece) (hx : p.board x = some q)
    (hc : q.color = p.turn)
    (hnp : q.kind = PieceKind.pawn → ¬ ((p.turn = Color.white ∧ x.2.val = 6) ∨ (p.turn = Color.black ∧ x.2.val = 1)))
    (h : movesOf p x y = true) :
    ∃ m ∈ legalMoves p, m.source = x ∧ m.destination = y ∧
      ∀ f : Position → Nat, afterMove p (some x) (some y) f = f (applyMove p m) := by
  obtain ⟨m, hm, hs, hd⟩ := movesOf_legal p x y q hx hc h
  have hk : queenOnly m = true := queenOnly_of_not_promoting p m hm q (by rw [hs]; exact hx) (by rw [hs]; exact hnp)
  exact afterMove_legal p x y m hm hs hd hk

/-- OUR STEP IN THE WORDS, the pawn-safe leg: `step_after` with "no pawn of ours at x on our seventh rank" for "no pawn of ours at x". -/
theorem step_after_np (c : Color) (g : Position → Bool) (p : Position) (n : Nat) (hc : p.turn = c) (x y : Square)
    (hown : count (fun u => piecesOf p c u && oset atSq (some x) u) ≥ 1)
    (hnp : count (fun u => kindSet p c PieceKind.pawn u && (oset atSq (some x) u && oset rankOf (some (named c ((0 : Fin 8), (6 : Fin 8)))) u)) ≤ 0)
    (hmv : count (fun u => oset (movesOf p) (some x) u && oset atSq (some y) u) ≥ 1)
    (H : ∀ m ∈ legalMoves p, (∀ f : Position → Nat, afterMove p (some x) (some y) f = f (applyMove p m)) →
      Ideas.forcedWithinB c g (applyMove p m) n = true) :
    Ideas.forcedWithinB c g p (n + 1) = true := by
  obtain ⟨q, hx, hqc, -⟩ := ours_at p c x hown
  have hq := not_promoting_at p c x hc hnp q hx hqc
  obtain ⟨m, hm, -, -, hafter⟩ := after_of_moves_np p x y q hx (by rw [hqc, hc]) hq (moves_of_count p x y hmv)
  exact step_ours c g p n hc m hm (H m hm hafter)

end Ideas.TreeStep
