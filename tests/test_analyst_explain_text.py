"""The explanation checker (catspace/incubator/analyst/explain_text.py): passes faithful text, flags planted errors."""
import json

import pytest

from catspace.io import paths
from catspace.incubator.analyst.explain_text import check

TEXT = ("Qxd6 grabs a pawn (Maia 36%). It fails to Qf2+ Kh2 Qxg2#, from +6.8 to mated. "
        "Instead Qxf8+ comes with check: after Kxf8, Rxh4 wins the queen on h4.")


def _facts():
    p = paths.derived("explain_facts_samples_v2.json")
    try:
        return {f["pid"]: f for f in json.load(open(p))}["1V7KO"]
    except (FileNotFoundError, KeyError):
        pytest.skip("the sample fact sheets are not on this machine (data/derived)")


def test_faithful_text_passes():
    assert check(TEXT, _facts()) == []


@pytest.mark.parametrize("wrong,kind", [("Qf2+ Kh2 Qxg2#", "move"), ("36%", "percent"), ("+6.8", "evaluation"), ("on h4", "square")])
def test_planted_errors_are_flagged(wrong, kind):
    planted = {"Qf2+ Kh2 Qxg2#": "Qe2+ Kh2 Qxg2#", "36%": "61%", "+6.8": "+3.1", "on h4": "on b7"}[wrong]
    bad = check(TEXT.replace(wrong, planted), _facts())
    assert any(k == kind for k, _ in bad), bad
