"""Tests for the stratified perfect-play pipeline (gen_stratified_perfect + train_stratified_field).

Fast/pure tests always run. Tablebase-dependent tests skip if data/syzygy is absent.
Covers: encode round-trips, piece-count derivation vs python-chess, the strata invariant
(captures reduce piece count and cannot be undone in one ply), the material-reachability mask,
the IQE quasimetric axioms (identity / non-negativity / triangle inequality), and perfect-play
label correctness vs the tablebase.
"""
import os
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from catspace.data.encode import board_from_packed, encode_meta, encode_packed
from experiments.gen_stratified_perfect import (ALL_MENUS, STRATA_MENUS, _emit_edges,
                                                negamax_tb, optimal_line, pcount)
from experiments.selfplay_generate import random_endgame_start
from experiments.train_stratified_field import count_vectors, reach_mask

SYZYGY = Path("data/syzygy")
needs_tb = pytest.mark.skipif(not SYZYGY.exists(), reason="no local syzygy tablebase")


# ---------------------------------------------------------------- pure / fast

def _sample(material, n, seed=0):
    rng = np.random.default_rng(seed)
    import experiments.selfplay_generate as sg
    sg._ENDGAME_MENUS.update(ALL_MENUS)
    out = []
    tries = 0
    while len(out) < n and tries < n * 200:
        tries += 1
        b = random_endgame_start(rng, material)
        if b is not None:
            out.append(b)
    return out


def test_encode_roundtrip_and_pcount():
    for b in _sample("krkbp", 20):
        pk, mt = encode_packed(b), encode_meta(b)
        b2 = board_from_packed(pk, mt)
        assert b2.board_fen() == b.board_fen()
        assert b2.turn == b.turn
        assert pcount(b2) == len(b.piece_map())


def test_piece_count_from_packed_matches_chess():
    boards = _sample("krrkbp", 30) + _sample("krk", 20)
    pk = np.stack([encode_packed(b) for b in boards])
    # count_vectors excludes kings; total pieces = 2 kings + sum(nonking counts)
    derived = 2 + count_vectors(pk).sum(1)
    truth = np.array([len(b.piece_map()) for b in boards])
    assert np.array_equal(derived, truth)


def test_capture_edges_reduce_piece_count_and_are_one_way():
    rng = np.random.default_rng(1)
    checked_capture = False
    for b in _sample("krkbp", 40, seed=2) + _sample("krrkbp", 40, seed=3):
        pc = pcount(b)
        for child, drop in _emit_edges(b, pc, edge_cap=99, rng=rng):
            cpc = pcount(child)
            if drop:
                assert cpc < pc, "a capture (drop) must reduce piece count"
                # one-way: material strictly decreased, so the PARENT board can never be
                # reproduced by any single legal move from the child (can't un-capture).
                for m in child.legal_moves:
                    c2 = child.copy(stack=False); c2.push(m)
                    assert c2.board_fen() != b.board_fen()
                checked_capture = True
            else:
                assert cpc == pc, "a non-capture must preserve piece count"
    assert checked_capture, "expected at least one capture edge in the sample"


def test_reach_mask_material_reachability():
    # cols: [Pw,Nw,Bw,Rw,Qw, Pb,Nb,Bb,Rb,Qb]
    A = np.array([[1, 0, 0, 2, 0, 1, 0, 1, 0, 0]])          # W: P + 2R ; B: P + B
    # identity is reachable
    assert reach_mask(A, A)[0]
    # cannot GAIN a white pawn
    B_gain_pawn = A.copy(); B_gain_pawn[0, 0] += 1
    assert not reach_mask(A, B_gain_pawn)[0]
    # promotion: white pawn -> white queen (pawn count drops, queen +1, within pawn budget)
    B_promo = A.copy(); B_promo[0, 0] -= 1; B_promo[0, 4] += 1
    assert reach_mask(A, B_promo)[0]
    # cannot add more non-pawn pieces than pawns available to promote
    B_two_q = A.copy(); B_two_q[0, 4] += 2                  # +2 queens but only 1 white pawn
    assert not reach_mask(A, B_two_q)[0]


def test_iqe_quasimetric_axioms():
    from catspace.nn.fb import TorchFB
    fb = TorchFB(d=32, channels=16, blocks=2, iqe=True, iqe_components=8, seed=0).eval()
    torch.manual_seed(0)
    X = torch.randn(24, 32) * 5.0
    with torch.no_grad():
        D = fb.distance_matrix(X, X).numpy()               # directed d(x_i -> x_j)
    assert (D >= -1e-4).all(), "non-negativity"
    assert np.allclose(np.diag(D), 0.0, atol=1e-3), "identity d(x,x)=0"
    # triangle inequality d(i->k) <= d(i->j) + d(j->k) (small float tolerance)
    tri = D[:, None, :] - (D[:, :, None] + D[None, :, :])   # [i,j,k]
    assert tri.max() <= 1e-3, f"triangle inequality violated by {tri.max():.4f}"


# ---------------------------------------------------------------- tablebase-dependent

@needs_tb
def test_perfect_play_labels_match_tablebase():
    from experiments.value_fixed_point import TB, white_pov_value
    tb = TB(str(SYZYGY))
    # KRk is a White win (White to move, mating material)
    krk = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    assert white_pov_value(krk, tb) == 1.0
    # K+B vs K is insufficient material -> draw
    kbk = chess.Board("4k3/8/8/8/8/8/8/2B1K3 w - - 0 1")
    assert white_pov_value(kbk, tb) == 0.5
    tb.close()


@needs_tb
def test_optimal_line_winner_sign_matches_value():
    from experiments.value_fixed_point import TB, white_pov_value
    tb = TB(str(SYZYGY))
    for b in _sample("krk", 8, seed=7):
        if b.turn != chess.WHITE:
            continue
        v = white_pov_value(b, tb)
        _, winner = optimal_line(b, tb)
        if v == 1.0:
            assert winner == 1, "White win must resolve to a White mate under perfect play"
        elif v == 0.5:
            assert winner == 0, "draw must not resolve to a mate"
    tb.close()


@needs_tb
def test_negamax_grounds_below_frontier():
    from experiments.value_fixed_point import TB, white_pov_value
    tb = TB(str(SYZYGY))
    checked = 0
    for b in _sample("krkp", 16, seed=11):
        wp = white_pov_value(b, tb)
        if wp is None:                                     # on-demand TB coverage gap -> skip
            continue
        v, grounded = negamax_tb(b, tb, depth=2, alpha=-1e9, beta=1e9, budget=[20000])
        assert grounded, "a TB-covered <=6-piece position must ground at the leaf"
        stm = wp if b.turn == chess.WHITE else (1.0 - wp)
        assert (v > 0.5) == (stm == 1.0)                   # win agrees with the tablebase
        assert (v < -0.5) == (stm == 0.0)                  # loss agrees with the tablebase
        checked += 1
    assert checked >= 3, "expected several TB-covered positions to check"
    tb.close()
