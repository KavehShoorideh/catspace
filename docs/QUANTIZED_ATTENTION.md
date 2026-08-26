# Quantized attention over concepts

Kaveh, 2026-08-26: *"attention between concepts is for sure going to be necessary... is
there a quantized version of attention where my query and key become quantized? 'Who is
attacking me' and the answer is that piece."* And on scope: *"let's get Maia out of it
entirely... we just want to get the WDL of Stockfish using quantized attention."*

## The question this line answers

> Can an **L1-sparse linear readout** over **binary concepts enriched by quantized
> attention** reproduce Stockfish's WDL?

Nothing else. Maia, the future-kind auxiliary, the Koopman operator and the six-ply
chains are all removed — not abandoned, *disentangled*. They are separate questions and
they were making each iteration eight hours long.

## Architecture

```
board ──► trunk (opaque)
            │
            ▼
      LEVEL 1: LFQ ──────────────► binary concepts, ±1 per dim
            │                       per square / piece slot / board head
            ▼
      LEVEL 2: quantized attention over those tokens
            │   question:  hard 1-of-R choice   (discrete, straight-through)
            │   answer:    top-k                (sparse, rest exactly zero)
            │   result:    LFQ                  (bits, not a hidden vector)
            ▼
      relational concepts, also ±1
            │
            ▼
      L1-sparse linear readout ──► WDL (3-way softmax)
```

Every input to the readout is a bit. Everything between the trunk and the answer is
either a concept or a discrete question about concepts.

## Why constrain rather than inspect

Reading a soft attention map is the thing the field tried and partly retracted. Jain &
Wallace (2019) showed attention weights do not reliably indicate what a model used;
Meister et al. (2021) showed sparsity does not repair that — exact zeros make a map
*readable* without making it *faithful*.

So the design does not read an attention map, it constrains one, and then **tests** the
resulting claim. Because the tokens are named and the questions are discrete, a head is
a typed relation:

```
(piece 3) --[question 5]--> (square e4)
```

`relation_alignment()` scores each head's edges against a relation computed independently
by python-chess (attacks / defends / blocks), reporting precision and lift over the
relation's own density. "Question 5 means attacks" is then falsifiable, which is the
property the critique says raw attention lacks.

## What gets measured

Primary: **balanced accuracy and per-class recall** of WDL on a *stratified* held-out
set. Not raw accuracy — the corpus is 63/14/23 W/D/L, so predicting "win" scores 0.637
for free, and that skew already reversed one of this project's conclusions once.

Arms, all sharing the same trunk, data and schedule:
1. linear L1 readout on level-1 concepts only
2. linear L1 readout on level-1 + level-2 (quantized attention)
3. the same with an unconstrained attention layer, as the capacity ceiling

Arm 2 vs arm 1 is the value of attention. Arm 2 vs arm 3 is what the quantization costs.

Secondary, and the reason for the design: per-head **relation alignment** against
python-chess ground truth, and the question-usage histogram — a head that only ever asks
one question is a dead channel, and a head that spreads over all R is not a typed
relation. Both are visible.

## Then, separately

Once WDL works, the operator returns as its own question — and Kaveh's framing there has
changed too: *"we don't need the composability of Koopman. We just want to predict what
the eval is six plies down given these moves."* Direct per-ply heads, not a composable
linear operator, are the cheaper hypothesis and should be tried first.

## Future concepts, 1 and 2 ply

Kaveh: *"we probably want a one and two ply future concept space predictor in there with
EMA, which gives us the foundation for growing it later."*

DIRECT per-ply heads, not one operator applied repeatedly — his own framing, since
composability is not needed to answer "what is the eval N plies down this line". Adding
a 3-ply head is a dict entry; nothing must stay linear, compose, or have a spectrum.

```
(concepts, move) ──► head[1] ──► logits over every bit of the concepts 1 ply later
(concepts, move) ──► head[2] ──► ... 2 plies later
                       ▲
              targets: EMA copy of the encoder, stop-gradient
```

**The EMA is not optional.** "Make the future predictable from the present" has a trivial
solution — make the concepts constant. A stop-gradient target from a slowly-updated
encoder is the standard remedy, and this repo has already been bitten by getting that
wrong (a resume once hard-copied the target and destroyed its lag silently).

**Accuracy alone is meaningless here.** A move changes few bits, so a predictor that
COPIES its input scores very high. Measured on a realistic synthetic target: a copier
gets **0.918 raw accuracy and 0.000 lift**. Every number is therefore quoted as

```
lift = (accuracy - copy_accuracy) / (1 - copy_accuracy)
```

0.0 for a copier however high its raw score, 1.0 for a perfect predictor. The test suite
includes the copier as a negative control so the metric is known to distinguish them.

Also reported: `bits_changed`, the fraction of concept bits a move actually moves. If
that is tiny the task is trivial; if it is near half the concepts are not stable enough
to roll forward, and either reading is worth having before interpreting a lift.
