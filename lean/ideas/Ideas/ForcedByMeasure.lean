/-!
# Forcing a goal by a measure -- the method (Kaveh 2026-10-12)

"We need a notion of progress in any frame of any lemma ... anti-progress on one lemma, progress on another." The method,
for any game (any pieces, any goal): a frame is a set `S` of positions (us to move) with a MEASURE `μ`; if from every
position of the frame we can force -- against every reply -- the goal, or a position of the frame whose measure is lower,
within `k` of our moves (a TEMPO move is the case k > 1: the measure falls only later), then from every position of the
frame the goal is forced within `k · (μ + 1)` of our moves (`forced_by_measure`).

The game is abstract: `ours p q` our moves (p us to move, q they to move), `theirs q r` their replies, `goal q` the goal
reached by our move (mate). A position after our move where they have no reply and no goal (a stalemate) is no success.
The king-and-queen frame (rust/analyst_core/src/mating.rs) checks the hypothesis on the board, move by move: every move it
plays carries such a certificate (1,000 / 1,000 play-outs: 0 uncertified moves, 2026-10-12).
-/

namespace Ideas.Forced

variable {α : Type}

/-- REACH: from `p` (us to move) we force, within `n` of our moves and against every reply, the `goal` (on our move) or a
position (us to move) satisfying `target`. -/
def Reach (ours theirs : α → α → Prop) (goal target : α → Prop) : Nat → α → Prop
  | 0, p => target p
  | n + 1, p => target p ∨ ∃ q, ours p q ∧ (goal q ∨ ((∃ r, theirs q r) ∧ ∀ r, theirs q r → Reach ours theirs goal target n r))

variable {ours theirs : α → α → Prop} {goal : α → Prop}

/-- More moves never hurt. -/
theorem Reach.mono {target : α → Prop} : ∀ {n m : Nat} {p : α}, Reach ours theirs goal target n p → n ≤ m →
    Reach ours theirs goal target m p
  | 0, 0, _, h, _ => h
  | 0, _ + 1, _, h, _ => Or.inl h
  | _ + 1, 0, _, _, hle => absurd hle (by omega)
  | n + 1, m + 1, p, h, hle => by
    rcases h with h | ⟨q, hq, hg⟩
    · exact Or.inl h
    · refine Or.inr ⟨q, hq, ?_⟩
      rcases hg with hg | ⟨hex, hall⟩
      · exact Or.inl hg
      · exact Or.inr ⟨hex, fun r hr => Reach.mono (hall r hr) (by omega)⟩

/-- A target that is itself force-able: reaching it within `n`, then the goal within `j` from it, is the goal within
`n + j`. -/
theorem Reach.bind {T : α → Prop} {j : Nat} (hT : ∀ r, T r → Reach ours theirs goal (fun _ => False) j r) :
    ∀ {n : Nat} {p : α}, Reach ours theirs goal T n p → Reach ours theirs goal (fun _ => False) (n + j) p
  | 0, p, h => by simpa using hT p h
  | n + 1, p, h => by
    rcases h with h | ⟨q, hq, hg⟩
    · exact Reach.mono (hT p h) (by omega)
    · have e : n + 1 + j = (n + j) + 1 := by omega
      rw [e]
      refine Or.inr ⟨q, hq, ?_⟩
      rcases hg with hg | ⟨hex, hall⟩
      · exact Or.inl hg
      · exact Or.inr ⟨hex, fun r hr => Reach.bind hT (hall r hr)⟩

/-- THE METHOD: a frame `S` with a measure `μ` such that from every position of it we force, within `k` moves, the goal or
a position of the frame with a lower measure -- then from every position of the frame the goal is forced, within
`k · (μ p + 1)` of our moves. -/
theorem forced_by_measure (S : α → Prop) (μ : α → Nat) (k : Nat)
    (hprog : ∀ p, S p → Reach ours theirs goal (fun r => S r ∧ μ r < μ p) k p) :
    ∀ p, S p → Reach ours theirs goal (fun _ => False) (k * (μ p + 1)) p := by
  intro p hp
  suffices H : ∀ m, ∀ p, S p → μ p = m → Reach ours theirs goal (fun _ => False) (k * (m + 1)) p by
    exact H (μ p) p hp rfl
  intro m
  induction m using Nat.strongRecOn with
  | ind m ih =>
    intro p hp hm
    have h := hprog p hp
    rw [hm] at h
    have hT : ∀ r, (S r ∧ μ r < m) → Reach ours theirs goal (fun _ => False) (k * m) r := by
      rintro r ⟨hr, hlt⟩
      have := ih (μ r) hlt r hr rfl
      exact Reach.mono this (Nat.mul_le_mul_left k hlt)
    have := Reach.bind hT h
    have e : k + k * m = k * (m + 1) := by rw [Nat.mul_add, Nat.mul_one, Nat.add_comm]
    rw [e] at this
    exact this

/-- Forcing with no target left is forcing the goal: `Reach … (fun _ => False) n p` says that from `p` every line of ours
(against every reply) ends in the goal within `n` moves. -/
theorem reach_false_goal {n : Nat} {p : α} (h : Reach ours theirs goal (fun _ => False) n p) :
    ∃ q, ours p q ∧ (goal q ∨ ((∃ r, theirs q r) ∧ ∀ r, theirs q r → ∃ m < n, Reach ours theirs goal (fun _ => False) m r)) := by
  cases n with
  | zero => exact absurd h id
  | succ n =>
    rcases h with h | ⟨q, hq, hg⟩
    · exact absurd h id
    · refine ⟨q, hq, ?_⟩
      rcases hg with hg | ⟨hex, hall⟩
      · exact Or.inl hg
      · exact Or.inr ⟨hex, fun r hr => ⟨n, by omega, hall r hr⟩⟩

end Ideas.Forced
