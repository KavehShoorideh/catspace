"""The screening's race (pipeline.race / best_of): successive halving on the same boards, the best among the finalists."""
from catspace.incubator.analyst.discovery.pipeline import best_of, race


class FakeVerifier:
    """Candidate k holds board i when (i * 7 + k) % 10 < k -- a higher k holds more boards; records as the verifier's."""
    def __init__(self):
        self.calls = []

    def verify_many(self, hole, cands, boards):
        self.calls.append((len(cands), len(boards)))
        out = []
        for term, schema in cands:
            k = int(term)
            pb = [[int((int(b) * 7 + k) % 10 < k), 0, 1] for b in boards]
            held = sum(x[0] for x in pb)
            out.append(dict(hole="h", term=term, schema=schema, defender="tablebase", n=len(boards), mated=0, lost=len(boards) - held,
                            played=len(boards), certified=0, states=len(boards), why={}, examples=[], per_board=pb))
        return out


def test_the_race_halves_and_keeps_the_strong():
    ver = FakeVerifier()
    cands = [(str(k), "append") for k in range(10)]
    recs = race(ver, {}, cands, [str(i) for i in range(40)])
    assert [c[0] for c in ver.calls] == [10, 5, 3, 2]                  # the field halves each stage
    assert sum(c[1] for c in ver.calls) == 40                          # the stages tile the boards
    finalists = [r for r in recs if r["n"] == 40]
    assert sorted(r["term"] for r in finalists) == ["8", "9"]
    assert best_of(recs)["term"] == "9"


def test_a_lucky_start_never_outranks_a_finalist():
    lucky = dict(n=5, score=1.0, term="lucky")
    finalist = dict(n=40, score=0.3, term="finalist")
    assert best_of([lucky, finalist])["term"] == "finalist"
