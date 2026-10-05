"""THE IDEA LOOP (Kaveh 2026-10-05: "build the engineering loop first, and then we slot in different research pieces
to see which one works ... plus some level of human in the loop testing that decides whether the loop is going off
the rails or the loop is going fine").

A ROUND is one pass over six stages on a frozen input batch:

    frame      the positions where the player went wrong (dev puzzle misses, game losses)
    mine       guards over relation literals around them (lemma_gen)
    translate  guards -> candidate ideas in the handoff plan's schema (hypotheses, footprint, bounded claim)
    prove      a tier for each: unconditional / forcing (a Lean theorem, standard axioms, a clean bridge against
               exhaustive search) or conjecture (held-out evidence only)
    use        how each is wired into the player: closure (proved only), target, order, sentence
    measure    the fixed protocol: fire rates and the player against frozen baselines, paired by position

Every stage reads and writes the records of `contracts.py`; any stage can be swapped for a research piece without
touching the driver. Each round ends in a one-page LEDGER.md and a GATE decision by Kaveh (continue, rollback, swap).
The basis of ideas is append-only (`basis.py`).

    python -m catspace.incubator.analyst.loop.round --round N            # run a round (needs round N-1's gate)
    python -m catspace.incubator.analyst.loop.gate --round N --decision continue --note "..."

Layout: everything a round produces under data/derived/loop/round_NNN (not committed); the reviewable record --
LEDGER.md, inputs, promoted ideas, measurements, the gate -- under artifacts/loop/round_NNN, and the basis in
artifacts/loop/basis.jsonl (committed, so a round is one commit and a rollback is a revert). See docs/LOOP.md.
"""
