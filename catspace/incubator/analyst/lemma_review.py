"""WHO PROPOSES (Kaveh 2026-10-10: "make this modular -- someday the LLM could be a token-based API call, but right now
the LLM is you and me reviewing something together and agreeing ... on the next piece of vocabulary").

One interface, three proposers. The miner (lemma_miner.py) calls `propose(target, report)` when lemmas leave cases open:
  ReviewProposer   writes the open cases to a review file (artifacts/experiments/lemma_review_<target>.md): the cases the
                   current vocabulary cannot explain, the facts they share, the lemmas already found, and a template for
                   a new concept; Kaveh and Claude agree on the concept and write it into lemma_vocab.py with its
                   source; the next run picks it up (returns the concepts whose source names a review)
  ApiProposer      the same review, sent to a model behind an API -- a stub until there is one
  NoProposer       mining only
A proposal is VOCABULARY, never a lemma: the miner turns vocabulary into lemmas, the verifier accepts or rejects them.
"""
from __future__ import annotations

from typing import Protocol

from catspace.io import paths


class Proposer(Protocol):
    def propose(self, target: str, report: dict) -> list[str]:
        """-> names of NEW concepts (already registered in lemma_vocab.VOCAB) to add to the vocabulary"""
        ...


class NoProposer:
    def propose(self, target, report):
        return []


class ReviewProposer:
    """The review between Kaveh and Claude: a file to read together, and the concepts we wrote back."""

    def propose(self, target, report):
        from . import lemma_vocab
        from pathlib import Path
        out = Path(paths.experiment(f"lemma_review_{target}.md"))
        lines = [f"# Review: {target}", "",
                 f"Open cases: {report['open_count']} of {report['positive_count']} '{target}' cases (and "
                 f"{report['open_negative_count']} 'not {target}' cases) are explained by no lemma below.", "",
                 "## Lemmas found (zero exceptions, holdout checked)", ""]
        for l in report["lemmas"]:
            lines.append(f"- {' AND '.join(l['premise']) or '(always)'} => {l['conclusion']}  "
                         f"(support {l['support']}, holdout exceptions {l['holdout_exceptions']}, {l['proof']})")
        lines += ["", "## Open cases (sample)", "", "| FEN after the move | move | facts true |", "|---|---|---|"]
        for c in report["open_sample"]:
            lines.append(f"| `{c['fen']}` | {c['move']} | {', '.join(c['facts'])} |")
        lines += ["", "## The next concept (write it in lemma_vocab.py)", "",
                  "```python",
                  "@concept(\"<name: the highest-level idea that separates these cases>\",",
                  "         \"<plain-language definition>\", \"<position | move | material>\",",
                  "         \"review YYYY-MM-DD: Kaveh + Claude -- <which open cases it explains>\")",
                  "def _new(c): return ...   # c.board (defender to move), c.move, c.before",
                  "```", ""]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines))
        print(f"review written to {out}")
        return [n for n, c in lemma_vocab.VOCAB.items() if c.source.startswith("review")]


class ApiProposer:
    """A model behind an API reading the same review and returning a concept -- not built (no API yet)."""

    def propose(self, target, report):
        raise NotImplementedError("the API proposer is a placeholder: for now the review is Kaveh and Claude (ReviewProposer)")
