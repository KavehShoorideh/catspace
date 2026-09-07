"""subgames.py: lemmas, their coupling, and the resource ledger (what a piece holds; the opportunity cost of moving it)."""
import chess

from catspace.incubator.analyst.memo import Memo
from catspace.incubator.analyst.subgames import static_lemmas, components, multiplicity, holdings, holdings_lines, cost_of_move
from catspace.incubator.analyst.plans import plans_for, plan_text


def test_lemmas_and_components_on_a_game_position():
    b = chess.Board("r4rk1/2q1bppp/npp1pn2/2PpN3/3P4/P1N1P3/5PPP/R2Q1RK1 b - - 5 15")
    memo = Memo()
    lem = static_lemmas(b, memo)
    assert any(l.target == chess.C5 and l.status == "falls" and l.side == chess.BLACK for l in lem)
    c5 = next(l for l in lem if l.target == chess.C5 and l.side == chess.BLACK)
    assert {chess.B6, chess.E7, chess.A6, chess.D4} <= c5.participants and c5.footprint      # attackers, the defender, recorded reads
    comps = components(lem)
    assert sum(len(g) for g in comps) == len(lem) and len(comps) >= 1
    assert max(multiplicity(lem).values()) >= 2                                            # coupling: some piece sits in two lemmas


def test_the_blocker_holds_the_rook_and_the_defender_holds_the_pawn():
    # the rook ending after ...Kxh5: the b4 pawn shields the a4 rook from the c4 rook (a QUIET lemma: nothing attacks a4 yet)
    b = chess.Board("8/8/1p3p2/p6k/RPr5/P5K1/8/8 w - - 0 2")
    held = holdings(b, Memo())
    assert [(h.lemma.target, h.effect, h.loss) for h in held[chess.B4]] == [(chess.A4, "exposed", 5.0)]
    assert any(h.lemma.target == chess.A5 and h.effect == "exposed" for h in held[chess.B6])
    assert holdings_lines(b, chess.WHITE)[0] == "The white pawn on b4 holds the white rook on a4 (5)"


def test_a_plan_pays_for_what_its_piece_holds():
    # the f6 knight keeps Qxh7# off the board; the e4 rook hangs to it. The plan exists (a rook outweighs the h7 pawn
    # in the ledger) but its opportunity cost is the mate: it is reported as not playable and sorted last
    b = chess.Board("5r1k/6pp/5n2/8/4R3/7Q/5PP1/6KR b - - 0 1")
    memo = Memo()
    held = holdings(b, memo)
    assert any(h.effect == "mated" for h in held[chess.F6])
    broken = cost_of_move(b, chess.Move.from_uci("f6e4"), held, memo)
    assert any(h.effect == "mated" for h in broken) and sum(h.loss for h in broken) >= 100
    pls = plans_for(b, chess.BLACK, memo=memo, top=3)
    nxe4 = next(p for p in pls if p.steps and p.steps[0].san == "Nxe4")
    assert nxe4.cost >= 100 and "keeps off Qxh7 (mate): not playable" in plan_text(b, nxe4)
    assert pls.index(nxe4) == len(pls) - 1
    assert "keeps off Qxh7 (mate)" in holdings_lines(b, chess.BLACK, memo)[0]
    # a move that captures the lemma's own attacker breaks nothing: after 16.dxc5, Qxe5 leaves c6 but takes the knight that hit it
    b = chess.Board("r4rk1/2q1bppp/n1p1pn2/2PpN3/8/P1N1P3/5PPP/R2Q1RK1 b - - 0 16")
    memo = Memo()
    held = holdings(b, memo)
    assert any(h.lemma.target == chess.C6 for h in held.get(chess.C7, []))
    assert cost_of_move(b, chess.Move.from_uci("c7e5"), held, memo) == []
