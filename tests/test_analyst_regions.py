import chess

from catspace.incubator.analyst.regions import region, regions, relations, PAIR_RELATIONS


def test_cut_off_king_and_a_rook_no_one_can_reach():
    # Kaveh's endgame: Black to move; the rook on b7 keeps the black king on the back rank
    b = chess.Board("4k3/1R6/4K3/4P3/8/8/8/5r2 b - - 0 58")
    regs = regions(b)
    bk = regs[chess.E8]
    assert chess.popcount(bk.safe) == 7                                     # the back rank (and the rook on b7 by c8, the opponent passing)
    assert not (bk.safe & regs[chess.E5].safe)                              # the king's safe region does not meet the pawn's path
    assert bk.contested and all(att for k, att in bk.contested.values() if k == 1)    # the first squares beyond are attacked, and the attackers are named
    rel = relations(b, regs)
    i = PAIR_RELATIONS.index("safe_meets_safe")
    assert rel["pairs"][(chess.E8, chess.E5)][i] == 0 and rel["pairs"][(chess.E6, chess.E5)][PAIR_RELATIONS.index("safe_attacks")] == 1
    # the black rook can reach a square attacking the white rook safely (any_attacks); the black king cannot attack it safely
    assert rel["pairs"][(chess.F1, chess.B7)][PAIR_RELATIONS.index("any_attacks")] == 1


def test_pawn_region_is_its_path_and_promotion_squares():
    b = chess.Board("8/8/8/8/8/8/4P3/k6K w - - 0 1")
    r = region(b, chess.E2)
    assert r.steps[chess.E4] == 1 and r.steps[chess.E3] == 1 and r.steps[chess.E8] == 5
    assert r.promotion_squares() == chess.BB_SQUARES[chess.E8]
    # a pawn facing an enemy king: the last squares are contested, by the king
    b2 = chess.Board("4k3/8/8/8/8/8/4P3/7K w - - 0 1")
    r2 = region(b2, chess.E2)
    assert chess.E7 in r2.contested and r2.contested[chess.E7][1] == chess.BB_SQUARES[chess.E8]
