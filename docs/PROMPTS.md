# Prompts that catch SILENT failures

Derived from real bugs in this repo, 2026-08-23/24. Every one of these was found late
or by luck; each prompt is written to force the check that would have caught it early.

The organising fact: **the unit gates caught 100% of crashes and 0% of the silent
bugs.** A crash announces itself. A wrong dimension, an unmasked loss, or a missing
term from a paper produces *plausible numbers*, which is far worse — it gets reported
as a result.

Vague prompts get vague compliance. Each of these demands a specific artefact back:
a list, a number, a table. If the answer is prose, the check did not happen.

---

## 1. Before any training launch

> Before launching: list every tensor dimension in this model, where each one is
> DEFINED, and every place it is READ. Any dimension that appears as a literal in more
> than one file is a bug — show me the list and fix them. Then show me the block
> decomposition summing to d_state, computed from the live objects, not from memory.

**Caught (would have):** the lift constructed with `d_code=8, d_sq=16` while binary
used 3 and 24, so it sliced the board block as 256 dims when it was 96. Every
interaction token was misaligned and nothing crashed — 2400 dims is enough to slice
from. That was the *tenth* literal-width bug in two days and the first one live in a
run.

---

## 2. When I introduce any library primitive

> You just used a library method we have not used before. Fetch its source or paper.
> List every optional argument, and for each one state whether we need it and WHY NOT
> if we are omitting it. Then tell me what its loss/return values mean, including
> whether a negative value is expected.

**Caught (would have):** `LFQ.forward` takes a `mask` for variable-length sequences.
We passed none, so 32% of every move batch (padding) and 26% of piece slots (dead
pieces) were counted as real codes by the entropy term whose entire job is shaping
codebook usage. Also would have resolved the negative aux loss immediately instead of
being waved past.

---

## 3. When I implement a named method

> You are implementing a method with a name and a paper. Before writing code, fetch
> the source and give me the exact formula with every term labelled. Then, after
> writing it, show me the mapping from each term in the paper to each line in our
> code, and list what we deliberately left out.

**Caught (would have):** ELMo's scalar mix is `γ · Σ sⱼhⱼ` with per-layer
**layer normalisation** before weighting. We had neither. Missing γ costs
optimisation; missing the layer-norm is worse — layers differ by 1.40× in activation
scale here, so the learned weights would have been compensating for magnitude rather
than expressing preference, making the depth profile (the whole point of the
mechanism) report the wrong thing. It would never have surfaced as a bug.

---

## 4. When I change any count or size

> You changed a count. Find every other quantity that is compared to, subtracted
> from, or sliced against it, and show me each one still makes sense at the new value.
> Include anything that becomes zero-length or negative.

**Caught (would have):** board heads cut 64 → 32 while `slow_heads` stayed 32, so
*every* head became slow and the fast-head slice was empty. VICReg on an empty tensor
is NaN; it killed three launches and took three wrong hypotheses to find.

---

## 5. Whenever I report a number

> Is this number measured on the distribution the model actually serves, or on inputs
> you constructed? How many samples, and what is the standard error? Is any point in
> this comparison inferred rather than measured?

**Caught (would have):** "the sharp head's level is untethered" — from hand-built FENs
(the opening position minus a queen) that no real game reaches. On 5,903 real unseen
positions the head was *better* calibrated than the incumbent. Also "separation tracks
play strength", published from three points where only two had arena numbers and the
third was inferred; it was falsified within the hour.

---

## 6. Gates

> Does this gate mutate anything it measures? Does it assert an invariant that is true
> in EVERY mode the code runs in — fresh, resumed, binary, float — or only the one you
> happened to be testing? Name the modes it is valid in.

**Caught (would have):** U19b took a real optimizer step, so every gate after it
measured a model the suite itself had trained (contamination was 3.5× on one
measurement). U22 called `.zero_()` on the thing it was probing, wiping learned
weights. U19, U21 and U22 each asserted a fresh-start invariant that is false for a
resumed model — three instances of the same trap.

---

## 7. Any quantity the run depends on

> Is this a log line or a gate? If the run would produce plausible-looking garbage
> when this quantity is wrong, it must be a gate that fails, not a number printed
> among forty others.

**Caught (would have):** `[gko0] start: ... sf 0 maia 409,600` printed at launch. The
sharp head had no gradient at all and the material-sightedness loss was off. It
trained 1,000 steps that way while every other gauge stayed green, because a loss term
gated on absent data reports `0.000`, which reads identically to "converged".

---

## 8. When a loss reads exactly 0.000, or a metric is suspiciously round

> Find that term's gating condition in the source and prove the data satisfies it.
> Zero is not convergence; it is usually a switch that is off.

---

## 9. After any negative or surprising result

> Before you conclude: is the thing you are measuring actually connected? Show me the
> gradient reaching the parameter you think this trains.

**Caught (would have):** would have shortened the "operator is the bottleneck"
investigation considerably — the decisive test (feed the value head TRUE children,
see if it ranks them: +0.556 vs −0.028 for rolled) should have been the *first*
experiment, not the tenth.

---

## The one-line version

> Show me the arithmetic, on the real distribution, with the gradient attached — and
> tell me which of these numbers you inferred rather than measured.
