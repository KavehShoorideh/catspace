# Architecture: interfaces and what crosses the bottleneck

Kaveh, 2026-08-25: *"I want an abstract definition of the architecture so that it's
clear where the interfaces are and what's being fed into what."*

This is the contract, not a tour of the code. Every shape below is read off a real
checkpoint (`gko8a_step8000`), and the file is the place to check before adding a head:
**if your new module's input is not on this list, it does not get built.**

---

## The one invariant

> **Nothing crosses the bottleneck except concepts.**

The trunk is opaque by design. Everything downstream of quantisation is required to be
a function of discrete codes alone. That is what makes any downstream claim
interpretable, and it is the property that silently broke twice (`sf_av`, then
`move_logits`) before it was made visible at startup.

Every run prints its actual head inputs on line 3 of the log:

```
[gko0] head inputs -- move_head: [concepts | quantised move code]
       | sf_av: [concepts | quantised code]
       | WDL: Linear(concepts->3), shared by root and rolled states
```

If that line disagrees with this document, the document is wrong or the run is.

---

## Stages

```
  board (graph)
        │
        ▼
  ┌─────────────────────────────────────────────────────┐
  │ 1. TRUNK          graph net, 4 layers, d_model 128  │  OPAQUE. No head may read it.
  └─────────────────────────────────────────────────────┘
        │  h_sq, h_mv, pre_g   (continuous, pre-quantisation)
        ▼
  ┌─────────────────────────────────────────────────────┐
  │ 2. QUANTISER      LFQ, binary: every dim is ±1      │  ◄── THE BOTTLENECK
  └─────────────────────────────────────────────────────┘
        │  zq (2528)   q_mv (64)          discrete
        ├──────────────────────────────┐
        ▼                              ▼
  ┌───────────────────────┐    ┌──────────────────────────────────┐
  │ 3. READOUTS (linear)  │    │ 4. LIFT + OPERATOR (rolling)     │
  │    concepts → answer  │    │    concepts → za → K(m)·za       │
  └───────────────────────┘    └──────────────────────────────────┘
                                       │ rolled za
                                       └──► back into 3 (same heads)
```

---

## 1. Trunk — opaque

| | |
|---|---|
| what | graph network over squares + moves, 4 layers, `d_model` 128 |
| outputs | `h_sq`, `h_mv`, `pre_g` — continuous |
| who may read it | **the quantiser only**, plus `decor` (see Exceptions) |

Warm-started from `gcf4h`; with `--trunk-layers 4` the first three load and the fourth
initialises fresh.

## 2. Quantiser — the bottleneck

LFQ (lookup-free): every dimension is a sign, so a code *is* a bit pattern and the
vocabulary is 2^bits with no learned codebook to collapse.

| block | width | structure |
|---|---|---|
| board | 96 | 32 heads × 3 bits |
| squares | 1536 | 64 squares × 24 bits |
| pieces | 768 | 32 slots × 24 bits |
| edge histogram | 128 | distribution **over** discrete edge codes |
| **`zq` total** | **2528** | 2400 dims are literal codes (measured: 1.9–2.8 distinct values/dim) |
| `q_mv` | 64 | 32 books × 2 bits, per move |

## 3. Readouts — linear, concepts only

Every head is a **linear map of the concept block**. With binary inputs this is the best
possible *additive* model — a GAM or Neural Additive Model fits the identical function —
so anything a head cannot express is, precisely, an interaction.

| head | input | output | trained on |
|---|---|---|---|
| `wdl` | `z[:, :2528]` | 3 → softmax | SF WDL, at the root **and after rolling** |
| `sf_av` | `[concepts \| q_mv]` (2592) | 64 bins | per-move SF value |
| `move_head` | `[concepts \| q_mv]` | 1 per move | Maia policy |
| `fkind_head` | `[concepts \| q_mv]` | K kinds | future-kind labels |
| `av_roll` | `[za \| move emb]` | 64 bins | per-move value from rolled states |

`wdl` is **one head with two call sites**. Rolled concepts occupy the same first 2528
columns of `za`, so "rolled-forward concepts must give the WDL Stockfish gives" is true
by construction rather than by two heads agreeing.

Sparsity is imposed by **proximal soft-thresholding** after each optimiser step, not by
an L1 penalty — under an adaptive optimiser a penalty yields small-but-nonzero weights
forever (measured: 2528/2528 still nonzero). The achieved nonzero count is printed every
eval; it is a fact, not a setting.

## 4. Lift + operator — rolling

| | |
|---|---|
| `lift` | `(zq, q_mv) → za = [zq | φ]`, `d_aug` 2784, φ = 256 |
| φ | attention over the concept tokens — a function of concepts only |
| `ops` | `BasisOps`: `K(m,z) = A + Σₖ wₖ(m,φ)·UₖVₖᵀ`, 128 bases × rank 32 |
| state conditioning | `state_proj: 256 → 64`, reads the φ tail |

The lift is **not** in the value path. It exists so the operator has a space to act in,
and rolled states re-enter the readouts through the same linear heads.

---

## Exceptions — the only two things not covered by the invariant

1. **`decor` reads `pre_g`** (pre-quantisation). It is a *regulariser*: it decorrelates
   the concept heads so they do not learn the same thing. It cannot be computed on
   discrete codes (no gradient), and it feeds **no prediction**, so no information
   reaches an output around the bottleneck. Open for review.
2. **`av_roll` reads a move embedding** — a learned 64-d embedding of 19 *discrete* move
   descriptors (piece type, promotion, …). That is move identity, not board content.

Anything else reading pre-quantisation features is a bug.

---

## Adding a head — the checklist

1. Its input must be `zq`, `q_mv`, or a rolled `za` — nothing else.
2. It should be linear unless you can show the interaction it needs.
3. Add it to the startup audit line, so the claim is visible in every log.
4. If it consumes rolled states, it must be the *same* head that consumes encoded ones,
   or you have two value functions and no guarantee they agree.

---

## Appendix: where regularisation belongs (literature, 2026-08-25)

Kaveh asked whether the concept space needs its own penalty or whether sparsity belongs
only in the readout. The answer turns on one structural fact.

**A sparsity penalty on the concept space is inert here.** The concepts are LFQ-binary:
every dimension is ±1, always. An L1 on activations — the standard sparse-autoencoder
lever — is a constant with zero gradient on ±1 codes. The SAE literature's whole L1
apparatus, and its known pathology of activation shrinkage (which is why
[TopK SAEs](https://cdn.openai.com/papers/sparse-autoencoders.pdf), Gao et al. 2024,
drop the L1 penalty and clamp instead) simply does not transfer to a binary code.

What *does* regularise a binary concept space, and what we already run:
* **codebook usage entropy** — LFQ's own term, which is what stops the vocabulary
  collapsing onto a few codes;
* **cross-head decorrelation** (`decor`) — stops heads learning the same thing;
* **the quantiser itself** — the bottleneck is the capacity constraint.

So sparsity belongs in the **readout**, which is what selects *which* concepts a given
prediction uses. That matches Concept Bottleneck Models, where the canonical design is
exactly "a sparse, one-layer classifier" over the concepts
([Koh et al. lineage](https://arxiv.org/pdf/2407.03921)).

Two findings from that literature worth acting on later:
* weight sparsity can cause **over-reliance on a single concept**; the standard remedy
  is dropout on the concept outputs during readout training, which spreads the decision.
* transformers are **not** made interpretable in place. The field's answer is to put a
  concept bottleneck *before* the opaque part and read out linearly from it — which is
  this architecture ([time-series CBM transformers](https://arxiv.org/abs/2410.06070)).
  Reported there: "model performance remains mostly unaffected, while the model shows
  much improved interpretability", consistent with our measured tax of -0.023 accuracy.
