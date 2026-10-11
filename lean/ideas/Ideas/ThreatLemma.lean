import Chess
import Ideas.PathLemma
import Ideas.FrameLemma

/-!
# The one-move threat, in stopper form (Kaveh 2026-10-11)

"When proving in Lean, prove the ways in which it can be stopped -- what are the things that can stop this attack, what
are the things that are not going to stop it -- and that becomes useful information for Rust to use later on when
deciding what squares are necessary to check for the continual usefulness of a lemma."

THE TACTIC: a piece of ours attacks their piece on `s`; our king stands on `k`. The claim: next move we capture on `s`
for the exchange law's value (Ideas.ExchangeLaw.seeXB), whatever they reply. THE STOPPERS -- a reply of theirs that
(1) moves the piece on `s` (an escape),
(2) changes who attacks `s` (a defender arrives, our attacker is captured or blocked, a discovery onto `s`),
(3) changes who attacks `k` (a check, or a pin of our attacker),
(4) captures elsewhere (the intermezzo: priced by the exchange law's composed tier, Ideas.ExchangeLaw.composedValue),
(5) promotes (a pawn on its seventh rank).
Each is read from named squares. (1)-(3) are read from the squares that BEAR on `s` and on `k` -- the theorem below:
a reply that touches none of them leaves the piece on `s`, every attack onto `s` and every attack onto `k` exactly as
they were, so the exchange on `s` (a function of the pieces on `s` and of who attacks it, with the x-rays behind them
on the same lines), and the legality of our capture (a function of who attacks `k`) are unchanged. (4) is read from
the index of their captures, (5) from their pawns on the seventh rank: squares Rust compares on its own. So the
dependency of a stored verdict is: the pieces on `s`, `k` and the escape squares; the attackers (first and x-ray) of
`s` and `k`; the pieces that can move to a square bearing on `s` or `k` (the one-step set: the arrivals of (2) and
(3)); the index; the seventh-rank pawns. rust/analyst_core/src/lemma.rs evaluates the stoppers and records exactly
those squares; plan.rs inputs_differ compares them, with the attack sets on the real occupancy (the tight form of
`reachB`, which is the empty-board form: an attack is a piece on a bearing square and a clear line, nothing else --
Ideas.FrameLemma.attack_reach, between_reach).
-/

namespace Ideas.ThreatLemma

open Chess
open Ideas.PathLemma
open Ideas.FrameLemma

/-- The stopper squares of the threat on `s` with our king on `k`: whatever bears on either. -/
def stopperB (s k x : Square) : Bool := reachB s x || reachB k x

/-- THE STOPPER THEOREM: a reply touching no stopper square leaves the piece on `s`, the piece on `k`, every attack onto
`s` and every attack onto `k` unchanged. -/
theorem threat_stoppers (p : Position) (m : Move) (s k : Square)
    (h : ∀ x, x ∈ touched p m → stopperB s k x = false) :
    (applyMove p m).board s = p.board s ∧ (applyMove p m).board k = p.board k ∧
    (∀ c a, pieceAttacksB (applyMove p m) c a s = pieceAttacksB p c a s) ∧
    (∀ c a, pieceAttacksB (applyMove p m) c a k = pieceAttacksB p c a k) := by
  have hF : ∀ t, (t = s ∨ t = k) → ∀ x, x ∈ touched p m → reachB t x = false := by
    intro t ht x hx
    have hs := h x hx
    simp only [stopperB, Bool.or_eq_false_iff] at hs
    rcases ht with rfl | rfl
    · exact hs.1
    · exact hs.2
  have ⟨hb, ha⟩ := frame_unchanged p m (fun t => t = s ∨ t = k) hF
  exact ⟨hb s (Or.inl rfl), hb k (Or.inr rfl), fun c a => ha s (Or.inl rfl) c a, fun c a => ha k (Or.inr rfl) c a⟩

/-- The corollary for the king: whether `k` is attacked by either side is unchanged -- no new check, no new pin line. -/
theorem king_attacked_unchanged (p : Position) (m : Move) (s k : Square)
    (h : ∀ x, x ∈ touched p m → stopperB s k x = false) (c : Color) :
    isAttackedB (applyMove p m) c k = isAttackedB p c k := by
  have hF : ∀ t, (t = s ∨ t = k) → ∀ x, x ∈ touched p m → reachB t x = false := by
    intro t ht x hx
    have hs := h x hx
    simp only [stopperB, Bool.or_eq_false_iff] at hs
    rcases ht with rfl | rfl
    · exact hs.1
    · exact hs.2
  exact attacked_unchanged p m (fun t => t = s ∨ t = k) hF k (Or.inr rfl) c

end Ideas.ThreatLemma
