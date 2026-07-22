# Conditional (covariate-gated) Sparse Autoencoders

A small, backward-compatible extension of [`dictionary_learning`](https://github.com/decoderesearch/SAELens)
that conditions the SAE dictionary on a per-example **covariate vector**. It subclasses the library's
`AutoEncoderTopK` and `TopKTrainer`, so it plugs into the existing `trainSAE` loop unchanged.

## Motivation — what a standard SAE can't do, and what this adds

A standard SAE learns **one global dictionary**: every atom is a feature of the entire input
distribution. But in many domains a concept's *meaning or relevance is context-dependent* — it is
informative in one sub-population and pure noise in another:

- **Chess (our setting).** The *bishop-pair advantage* is decisive in **open** endgames and nearly
  irrelevant in **closed** openings; *king safety* only matters while there are pieces to attack.
- **Language.** A latent may read differently across registers, languages, or topics.

A global SAE **averages over contexts**, with two consequences: (1) a context-specific concept is
*washed out* — too rare globally to earn its own atom — or *smeared* across several atoms; and (2)
there is no way to recover each concept's **domain of applicability** (the contexts where it applies).

Conditioning the dictionary on observable covariates `c` fixes both:

1. **Context-specific concepts get their own atoms.** A FiLM gate `sigmoid(g(c)) ∈ (0,1)^dict`
   multiplies the pre-top-k activations, so the SAE can allocate dictionary capacity *per context*.
   In the chess value field, king-safety atoms localize to castled-king contexts and a bishop atom
   localizes to open positions — concepts a global SAE reported only as low-correlation "novel" atoms
   even though their prevalence in the atom's own cluster was ~2–3× baseline.
2. **Each atom carries its domain, for free.** Read the gate `g(c)` against the covariates and you
   get, per atom, which contexts it fires in — its domain of applicability.

It is a **strict generalization**: with `cond_dim = 0`, or once a gate saturates open, it reduces to
the standard TopK SAE, so adding it changes no existing behavior.

## API

```python
from contrib.dl_conditional import ConditionalAutoEncoderTopK, ConditionalTopKTrainer

# Dictionary: encode gains an optional `cond`; cond=None == the base AutoEncoderTopK
ae = ConditionalAutoEncoderTopK(activation_dim=D, dict_size=M, k=K, cond_dim=C)
code = ae.encode(x, cond=c)          # relu(W_enc (x - b_dec)) * sigmoid(gate(c)), then top-k
x_hat = ae.decode(code)

# Trainer: activations arrive as [x | cond] (last cond_dim columns are the covariates)
tr = ConditionalTopKTrainer(steps=S, activation_dim=D, dict_size=M, k=K, cond_dim=C,
                            layer=0, lm_name="my_model", device="cuda")
for step in range(S):
    tr.update(step, torch.cat([x_batch, c_batch], dim=1))   # standard trainSAE loop works too
```

**Convention.** The trainer expects each activation row as `[x | cond]` (the covariates appended). A
buffer that appends covariates to activations is all `trainSAE` needs; the reconstruction objective
(L2 + auxk dead-atom revival) is computed on `x` only. `b_dec` is initialized from the `x` part.

## Upstream note

`TopKTrainer` builds its dictionary as `dict_class(activation_dim, dict_size, k)` with no hook for
extra constructor args, so `ConditionalTopKTrainer` re-instantiates the AE + optimizer once. A
one-line, backward-compatible `dict_class_kwargs: dict = {}` on the base trainer would make this a pure
subclass — worth proposing together with this feature.

## Files
- `conditional_topk.py` — `ConditionalAutoEncoderTopK`, `ConditionalTopKTrainer`.
- Used by `experiments/conditional_sae_dl.py` in this repo (concept discovery on the chess value field).
