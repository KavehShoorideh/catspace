"""The Rust core must agree with the Python domain exactly: outcomes (one reply and the episode, absolute and
relative) and every base relation, on random placements and random real positions."""
import random

import chess
import pytest

analyst_core = pytest.importorskip("analyst_core")
from catspace.incubator.analyst import chess_domain as cd
from catspace.incubator.analyst import lemma_gen as lg


def _random_states(n, seed=0):
    rng = random.Random(seed); out = []
    types = cd.TYPES
    while len(out) < n:
        k = rng.randint(1, 3)
        ts = tuple(rng.choice(types) for _ in range(k))
        for st, _ in cd.placements(ts, 3, rng.randrange(10 ** 6)):
            out.append(st)
            if len(out) >= n:
                break
    return out


@pytest.mark.parametrize("depth,relative", [(1, False), (1, True), (3, False), (0, True)])
def test_outcomes_agree_with_python(depth, relative):
    dom = cd.domain(oracle="frame", depth=depth, shapes=False)
    g = lg.Generator(dom, cap=3, log=lambda *a, **k: None, relative=relative, sequential=False)
    n = 0
    for st in _random_states(60, seed=depth * 7 + int(relative)):
        e = st.ids[0]
        base = None
        if relative:
            base = dom.null_reply_scalar(st, e)
            if base is None:
                continue
        py = {}
        for a in dom.actions(st, e):
            after = dom.transition(st, a)
            py[a[1].uci()] = dom.reply_scalar(after, e) - (base if base is not None else dom.scalar(st, e))
        b = st.board.copy(stack=False); b.turn = b.piece_at(e).color
        rs = dict(analyst_core.outcomes(b.fen(), list(st.ids), e, depth, relative))
        assert set(rs) == set(py), (b.fen(), set(rs) ^ set(py))
        for k in py:
            assert abs(rs[k] - py[k]) < 1e-9, (b.fen(), k, rs[k], py[k])
        n += 1
    assert n >= 30


def test_base_relations_agree_with_python():
    dom = cd.domain(oracle="frame", shapes=False)
    rels = {r.name: r for r in dom.relations}
    rows = []
    for st in _random_states(80, seed=3):
        e = st.ids[0]; b = st.board.copy(stack=False); b.turn = b.piece_at(e).color
        acts = list(dom.actions(st, e))
        for a in acts[:4]:
            rows.append((st, a))
    lits = []
    for name, r in rels.items():
        k = 3
        for sl in __import__("itertools").permutations(range(k), r.arity):
            lits.append((name, list(sl)))
    packed = [(st.board.fen().replace(" w ", " " + ("w" if st.board.piece_at(a[1].from_square).color else "b") + " ") if False else _fen_for(st, a), list(st.ids) + [-1] * (3 - len(st.ids)), a[1].uci()) for st, a in rows]
    masks = analyst_core.literal_masks(packed, lits)
    disagreements = 0; checked = 0
    for (name, sl), col in zip(lits, masks):
        r = rels[name]
        for i, (st, a) in enumerate(rows):
            if any(s >= len(st.ids) for s in sl):
                continue
            args = [st.ids[s] for s in sl]
            try:
                py = bool(r.fn(st, a, *args)) if r.on_action else bool(r.fn(st, *args))
            except Exception:
                py = False
            checked += 1
            if py != bool(col[i]):
                disagreements += 1
                assert False, f"{name}{sl} on {_fen_for(st, a)} {a[1].uci()}: python {py} rust {bool(col[i])}"
    assert checked > 1000


def _fen_for(st, a):
    b = st.board.copy(stack=False); b.turn = b.piece_at(a[1].from_square).color
    return b.fen()
