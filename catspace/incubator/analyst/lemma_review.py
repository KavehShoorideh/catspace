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
        lines = [f"# Review: {target}", "", f"Data: {report.get('source', '')}.", "",
                 f"Open: {report['open_count']} '{target}' cases and {report['open_negative_count']} 'not {target}' cases "
                 f"are decided by no accepted lemma.", ""]
        # THE CRITICAL CASE FIRST (argument-based machine learning: Mozina, Zabkar, Bratko 2007; on chess tablebases,
        # Guid et al.): the largest group of undecided cases sharing one fact pattern -- an argument for it is what the
        # next concept must capture
        lines += ["## The critical open cases (the largest groups sharing one fact pattern)", ""]
        for c in report.get("critical", []):
            lines.append(f"- **{c['group']} cases**, {target} = {c['target']}, e.g. `{c['fen']}` after {c['move']}; facts: {', '.join(c['facts'])}")
        lines += ["", "What argument explains this group? Write it as a concept below; the next run shows its "
                  "counterexamples before it is accepted.", "",
                  "## Rules with exceptions (each exception set is a candidate concept)", ""]
        for l in report.get("near", []):
            lines.append(f"- {' AND '.join(l['premise'])} => {l['conclusion']} UNLESS {l['exceptions']} of {l['support']}: "
                         + "; ".join(f"`{e['fen']}` after {e['move']}" for e in l.get("exception_cases", [])[:3]))
        lines += ["", "## Accepted lemmas (zero exceptions, holdout-checked, useful, each deciding a case no better one decides)", ""]
        for l in report["lemmas"]:
            lines.append(f"- {' AND '.join(l['premise']) or '(always)'} => {l['conclusion']}  "
                         f"(support {l['support']}, utility {l.get('utility_us', 0):.1f} us/context, {l['proof']})")
        lines += ["", f"Redundant (covered by better ones): {len(report.get('redundant', []))}; retired (cost more than "
                  f"they save): {len(report.get('retired', []))}.", "",
                  "## Open cases (sample)", "", "| FEN after the move | move | target | facts true |", "|---|---|---|---|"]
        for c in report["open_sample"]:
            lines.append(f"| `{c['fen']}` | {c['move']} | {c.get('target')} | {', '.join(c['facts'])} |")
        lines += ["", "## The next concept (write it in lemma_vocab.py)", "",
                  "```python",
                  "@concept(\"<name: the highest-level idea that separates these cases>\",",
                  "         \"<plain-language definition>\", \"<position | move | material | tactic>\",",
                  "         \"review YYYY-MM-DD: Kaveh + Claude -- <which open cases it explains>\")",
                  "def _new(c): return ...   # c.board (defender to move), c.move, c.before, c.labels",
                  "```", ""]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines))
        print(f"review written to {out}")
        return [n for n, c in lemma_vocab.VOCAB.items() if c.source.startswith("review")]


class ApiProposer:
    """A model behind an API reading the same review and returning a concept -- not built (no API yet)."""

    def propose(self, target, report):
        raise NotImplementedError("the API proposer is a placeholder: for now the review is Kaveh and Claude (ReviewProposer)")
