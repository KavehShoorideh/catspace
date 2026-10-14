"""THE DISCOVERY SYSTEM (Kaveh 2026-10-12). Three tracks, kept apart so they can be reasoned about:

  BASE        the vocabulary -- deterministic, interpretable primitives (Lean `Ideas.Base.Vocabulary`, Rust
              `vocab.rs`, here `base_vocabulary.py`); never learned, never generated; chosen with Kaveh
  HANDWRITTEN lemmas and measure components written by people (Lean `Ideas/*.lean` outside Base/ and Generated/, Rust
              `mating.rs` ...)
  GENERATED   what the pipeline found -- each with its term over the base, its hole, its verifier evidence (the registry
              `generated_registry.json`; Lean `Ideas.Generated.*` and Rust `generated.rs`, both written by `registry.py`,
              never by hand)

THE PIPELINE: proposers (several: enumerative "smaller first", a graph network, a transformer) each rank candidate terms
for a HOLE (a choice an operator must make: a measure component, a relaxation ...); a FILTER merges their rankings into a
budget; the VERIFIER checks every candidate the filter kept (certification on the hole's boards against every reply,
then whole games). Learned proposers only order the search -- meaning comes from the base, correctness from the verifier.
"""


def _install_terms():
    """The chess terms (compositions of the base words) into the evaluator, once per process."""
    try:
        from .base_vocabulary import install_terms
        install_terms()
    except ImportError:                                  # (the extension not built: nothing to install into)
        pass


_install_terms()
