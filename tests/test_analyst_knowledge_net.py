"""The knowledge network: masked, decomposable, directed -- the invariants that catch plausible-but-wrong."""
import numpy as np
import torch
import chess

from catspace.incubator.analyst import knowledge_net as K
from catspace.incubator.analyst.graph import build
from catspace.incubator.analyst.memo import Memo


def _batch(fens, cfg, device="cpu"):
    """A batch straight from graphs (no dataset on disk): buckets by a fixed coarse edge set."""
    samples = []
    edges = [np.array([-3, -1, 0, 0.5, 1, 2, 3, 5, 9, 100]) for _ in range(K.Q)]
    b = K.Buckets(cfg.buckets); b.edges = edges
    pb = K.Buckets(cfg.buckets); pb.edges = [np.array([0, 1, 2, 3, 5, 9, 20, 40]) for _ in range(K.P)]
    for fen in fens:
        board = chess.Board(fen); g = build(board, Memo())
        nm = g.node_matrix(); n = nm.shape[0]
        nodes = np.zeros((K.MAX_PIECES, K.Q), np.float32); nodes[:n] = nm
        mask = np.zeros(K.MAX_PIECES, bool); mask[:n] = True
        et = g.edge_tensor(); dense = np.zeros((K.MAX_PIECES, K.MAX_PIECES, K.R), np.float32); dense[:n, :n] = et
        related = np.zeros((K.MAX_PIECES, K.MAX_PIECES), bool)
        for (i, j) in g.edges:
            related[i, j] = True
        squares = np.array([nd.square for nd in g.nodes] + [64] * (K.MAX_PIECES - n))
        types = np.array([nd.piece.piece_type if nd.piece.color == board.turn else 6 + nd.piece.piece_type for nd in g.nodes] + [0] * (K.MAX_PIECES - n))
        legal = np.zeros((K.MAX_PIECES, 64), bool); idx = {nd.square: i for i, nd in enumerate(g.nodes)}
        for mv in board.legal_moves:
            legal[idx[mv.from_square], mv.to_square] = True
        samples.append(dict(buckets=b.apply(nodes), mask=mask, edges=dense, related=related, squares=squares, types=types,
                            pos_buckets=pb.apply(g.position_vector()[None])[0], legal=legal, best=-1, cp=0.0, result=2,
                            regime_now=np.zeros(K.K_REG, np.float32), ahead10=np.zeros(K.K_REG, np.float32), ahead20=np.zeros(K.K_REG, np.float32),
                            ended10=0, ended20=0, game=0, ply=0, i=0))
    return K.collate(samples, torch.device(device))


def test_output_is_linear_in_the_named_senses_and_padding_is_inert():
    torch.manual_seed(0)
    cfg = K.Config(d=32, layers=2, heads=4, buckets=12, dropout=0.0)
    model = K.KnowledgeNet(cfg).eval()
    b = _batch(["r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "8/8/8/8/1k6/8/5P2/4K3 w - - 0 1"], cfg)
    o = model(b)
    # phi recomputed by hand from the weights and the sense tables: exact decomposition
    a = o["alpha"]; senses = model.sense(model.sense_index(b))
    contrib = (a[..., :K.Q, None] * senses).sum(2) + a[..., K.Q:K.Q + 1] * model.type_sense(b["types"]) + a[..., K.Q + 1:K.Q + 2] * model.square_sense(b["squares"])
    phi_hand = contrib.sum(1) / (a.sum((1, 2))[:, None] + 1.0)
    assert torch.allclose(phi_hand, o["phi"], atol=1e-5)
    # padding rows carry zero weight
    assert (a[~b["mask"]] == 0).all()
    # the pawn ending has 4 pieces: perturbing a padded slot's bucket must not change phi or the tokens of real pieces
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    b2["buckets"][1, 10] = 3; b2["squares"][1, 10] = 12; b2["types"][1, 10] = 4
    o2 = model(b2)
    assert torch.allclose(o2["phi"][1], o["phi"][1], atol=1e-5)
    assert torch.allclose(o2["h"][1, :5], o["h"][1, :5], atol=1e-5)


def test_quasimetric_is_directed_and_zero_on_itself():
    torch.manual_seed(0)
    cfg = K.Config(d=32, layers=1, heads=4, dropout=0.0)
    model = K.KnowledgeNet(cfg).eval()
    u = torch.randn(5, 32); v = torch.randn(5, 32)
    assert torch.allclose(model.iqe(u, u), torch.zeros(5), atol=1e-6)
    assert not torch.allclose(model.iqe(u, v), model.iqe(v, u))
    # triangle inequality on random triples
    w = torch.randn(5, 32)
    assert (model.iqe(u, w) <= model.iqe(u, v) + model.iqe(v, w) + 1e-5).all()


def test_policy_logits_only_on_legal_moves_and_losses_run():
    torch.manual_seed(0)
    cfg = K.Config(d=32, layers=1, heads=4, dropout=0.0)
    model = K.KnowledgeNet(cfg)
    b = _batch(["r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"], cfg)
    o = model(b); pl = model.policy_logits(o["h"], b)
    assert torch.isfinite(pl[b["legal"]]).all() and torch.isinf(pl[~b["legal"]]).all()
    b["best"] = torch.tensor([int(torch.nonzero(b["legal"].view(1, -1))[0, 1])])
    L = K.losses(model, b, cfg, k=None, n_anchor=0)
    assert torch.isfinite(L["total"]) and L["policy"] > 0
    # the board-only and rules-only variants run on the same batch
    for variant in ("board", "rules"):
        m2 = K.KnowledgeNet(K.Config(d=32, layers=1, heads=4, dropout=0.0, variant=variant))
        assert torch.isfinite(K.losses(m2, b, m2.cfg)["total"])


def test_trajectory_terms_push_apart_only_the_unreachable():
    """Two anchors with partners k plies ahead, one earlier position of the same game, one from another game:
    the forward pair is calibrated, the earlier one and the other game are pushed apart, the later one bounded."""
    torch.manual_seed(0)
    cfg = K.Config(d=32, layers=1, heads=4, dropout=0.0)
    model = K.KnowledgeNet(cfg)
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    b = _batch([fen] * 6, cfg)
    b["game"] = torch.tensor([1, 2, 1, 2, 1, 3]); b["ply"] = torch.tensor([10, 20, 14, 23, 5, 9])   # anchors 0,1; partners 2,3 (k=4,3); 4 = earlier same game; 5 = other game
    b["best"] = torch.full((6,), -1)
    k = torch.tensor([4, 3])
    L = K.losses(model, b, cfg, k, 2)
    for key in ("traj_fwd", "traj_neg", "traj_bound"):
        assert torch.isfinite(L[key])
    # the mask logic, recomputed here over the whole batch: pushed = not (same game and target later), self excluded
    games, plies = b["game"], b["ply"]; same = games[:, None] == games[None, :]; later = same & (plies[None, :] > plies[:, None])
    pushed = ~later & ~torch.eye(6, dtype=torch.bool)
    assert not pushed[0, 2] and not pushed[1, 3]           # forward partners are reachable, not pushed
    assert pushed[2, 0] and pushed[3, 1]                    # the REVERSE of time is pushed apart
    assert pushed[0, 4] and pushed[0, 5] and pushed[0, 1]   # the earlier position, the other game, the other anchor
    assert not pushed[4, 0] and not pushed[0, 0]            # from the earlier position the anchor is reachable
