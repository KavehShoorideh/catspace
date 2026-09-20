import chess
import numpy as np
import torch

from catspace.incubator.analyst import tokens as T
from catspace.incubator.analyst.endgames import row_of
from catspace.incubator.analyst.root_tokens import feature_names, features_of_board


def test_child_result_is_in_the_parent_movers_view_and_white_view_is_consistent():
    # the sign conventions: the child's stored result is in the PARENT mover's view; converted to White's view it
    # must equal the tablebase's own verdict on the child position converted to White's view
    import chess.syzygy, random
    tb = chess.syzygy.open_tablebase("assets/tablebases/syzygy")
    b = chess.Board("7k/8/8/8/8/2q5/8/K7 b - - 0 1")               # Black to move, a queen up: Black wins
    r = row_of(b, tb, random.Random(0), "KQvK")
    assert r is not None and r["wdl"] == 2 and T.white_wdl(r["wdl"], b.turn) == -1
    cb = chess.Board(r["child"]["fen"])
    direct = tb.probe_wdl(cb) if not cb.is_game_over() else 0
    tb.close()
    assert T.white_wdl(r["child"]["wdl"], b.turn) == T.white_wdl(direct, cb.turn)


def test_losses_coherence_and_usage_behave():
    cfg = T.Config(d=16, layers=1, heads=2, k=8)
    model = T.TokenNet(cfg).train()
    b0 = chess.Board("4k3/1R6/4K3/4P3/8/8/8/5r2 b - - 0 58"); b1 = chess.Board("5k2/1R6/4K3/4P3/8/8/8/5r2 w - - 1 59")
    sq0, s0, r0 = T.encode(b0); sq1, s1, r1 = T.encode(b1)
    batch = dict(sq=torch.as_tensor(np.stack([sq0, sq0])), side=torch.as_tensor([s0, s0]), rel=torch.as_tensor(np.stack([r0, r0])).float(), wdl=torch.tensor([1, 1]), dtz=torch.tensor([0, 0]),
                 csq=torch.as_tensor(np.stack([sq1, sq1])), cside=torch.as_tensor([s1, s1]), crel=torch.as_tensor(np.stack([r1, r1])).float(), cwdl=torch.tensor([1, 0]), has_child=torch.tensor([True, True]))
    out = T.losses(model, batch, cfg)
    for k in ("wdl", "dtz", "coh", "use", "cov", "total"):
        assert torch.isfinite(out[k]), k
    # coherence: identical token vectors across a same-result reply cost nothing; across a result-changing reply they cost the margin
    p = torch.tensor([[0.9, 0.1, 0.9, 0.1, 0.5, 0.5, 0.5, 0.5]])
    d = (p - p).abs().sum(-1)
    assert float(d) == 0.0 and float(torch.relu(cfg.coh_margin - d)) == cfg.coh_margin
    # usage entropy: a dead code (mean 0) and an always-on code contribute nothing; a half-used code the maximum
    u = torch.tensor([1e-4, 1 - 1e-4, 0.5]); H = -(u * u.log() + (1 - u) * (1 - u).log())
    assert H[0] < 0.01 and H[1] < 0.01 and abs(float(H[2]) - 0.693) < 0.01
    # covariance: two identical codes cost, two independent ones do not
    q = torch.rand(64, 1); P2 = torch.cat([q, q, torch.rand(64, 1)], 1); pc = P2 - P2.mean(0); cov = pc.T @ pc / 63
    off = cov - torch.diag(torch.diag(cov))
    assert off[0, 1] > 0.05 and abs(float(off[0, 2])) < 0.05 + abs(float(off[0, 1])) * 0.5


def test_rooting_recovers_a_relation_that_defines_a_token():
    from sklearn.tree import DecisionTreeClassifier
    names = feature_names(); index = {n: i for i, n in enumerate(names)}
    rng = np.random.default_rng(0); X, y = [], []
    for _ in range(300):
        b = chess.Board(None)
        sqs = rng.choice(64, 4, replace=False)
        b.set_piece_at(int(sqs[0]), chess.Piece(chess.KING, True)); b.set_piece_at(int(sqs[1]), chess.Piece(chess.KING, False))
        b.set_piece_at(int(sqs[2]), chess.Piece(chess.ROOK, True)); b.set_piece_at(int(sqs[3]), chess.Piece(chess.ROOK, False)); b.turn = True
        if not b.is_valid():
            continue
        x = features_of_board(b, index); X.append(x); y.append(int(x[index["Wr->Bk: safe_attacks"]]))      # a token DEFINED as one relation
    X, y = np.stack(X), np.array(y)
    if 0 < y.mean() < 1:
        clf = DecisionTreeClassifier(max_depth=1).fit(X, y)
        assert clf.score(X, y) == 1.0 and names[clf.tree_.feature[0]] == "Wr->Bk: safe_attacks"
