#!/usr/bin/env python
"""eval_winning_track.py -- following our field, do we stay on the winning track?

"there should be a bunch of moves that stockfish says maintains your winning chances, and then
some moves that basically lose the winning chances. I wanna know if by following our best
guidance down the field, do we pick any of those losing moves, or do we stay on the winning
track? how often is the top move given by our field inside of the winning track?" (Kaveh
2026-08-18)

A far better question than exact-move agreement. Most positions have several acceptable moves, so
matching Stockfish's single choice punishes a field for picking a different good move. What
actually matters is whether it picks a move that THROWS THE RESULT AWAY.

METHOD. Stockfish scores EVERY legal move (multipv = all), giving a win probability per move from
the mover's side. The track is defined on OUTCOME CATEGORY, not centipawns, so the bar is
"did you keep the result" rather than "did you find the best move":

    category(wp)  WIN wp >= 0.75 | DRAW 0.25 < wp < 0.75 | LOSS wp <= 0.25
    on-track      the move does not DROP the category the best move achieves

Only positions where the distinction EXISTS are scored -- at least one on-track and at least one
off-track move. Everywhere else the question is vacuous and including it would inflate every
method equally.

BASELINES on identical positions: a random legal move, and material-greedy. Stockfish's own top
move is on-track by construction and is not reported as a result.

    .venv/bin/python -m ...eval_winning_track --model csg --ckpt artifacts/experiments/csg1_latest.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch


def wp_of(score, mover_pov=True):
    """python-chess PovScore -> win probability for the side to move."""
    s = score.pov(score.turn) if hasattr(score, "turn") else score
    return s.wdl().expectation() if s.wdl() is not None else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["csg", "cqm", "cef", "bc", "gdv"], default="csg")
    ap.add_argument("--lam", type=float, default=1.0,
                    help="gdv only: dV weight in zs(prior) + lam*zs(dV), matching the arena")
    ap.add_argument("--ckpt", default="artifacts/experiments/csg1_latest.pt")
    ap.add_argument("--concept-ckpt", default="artifacts/experiments/cef3_step10000.pt")
    ap.add_argument("--codes", default="data/derived/concept_codes_cef3_6545146.npz")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--engine", default="/opt/homebrew/bin/stockfish")
    ap.add_argument("--win-hi", type=float, default=0.75)
    ap.add_argument("--win-lo", type=float, default=0.25)
    ap.add_argument("--n-poles", type=int, default=96, help="cqm only: poles sampled per class")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import chess
    import chess.engine
    from catspace.research.tools.training_infra.concepts_reference import (
        TOK_PIECES, board_from_row)
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_event_field import ConceptEvent
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_capture_field import factor_vocab
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    dev = args.device
    ck = torch.load(args.concept_ckpt, map_location="cpu", weights_only=False)
    A = ck["args"] if isinstance(ck.get("args"), dict) else vars(ck["args"])
    tr = T.build(n_human=0, n_sf=A["games"], seed=A["seed"], cache=True,
                 max_plies=A["max_plies"], n_piecedown=A["n_piecedown"])
    n = len(tr.tok)
    lz = np.load(paths.derived(f"tb_entry_5-4-3men_{n}.npz"), allow_pickle=True)
    vocab = [str(x) for x in lz["vocab"]]
    cr, ce, co, egs = factor_vocab(vocab)
    enc = ConceptEvent(d_model=A["d_model"], heads=A["heads"], codes=A["codes"],
                       square_codes=A["square_codes"], hidden=A["hidden"],
                       n_res=len(vocab), factored=True, n_eg=len(egs),
                       cls_reach=cr, cls_eg=ce, cls_out=co).to(dev)
    enc.load_state_dict(ck["state_dict"]); enc.eval()

    @torch.no_grad()
    def enc_np(tk, gb):
        z, ids, ids_sq, _ = enc.encode(torch.from_numpy(tk.astype(np.int64)).to(dev),
                                       torch.from_numpy(gb.astype(np.float32)).to(dev))
        return z, ids, ids_sq

    gdv_score_moves = None
    if args.model == "gdv":
        from catspace.research.components.encoder.approaches.reach_probability.experiments \
            .graph_delta_value import GraphDeltaValue, batch_of
        from catspace.research.tools.training_infra.gen_move_graphs import (
            parent_graph, glob_of_board)
        gk = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        GA = gk["args"] if isinstance(gk.get("args"), dict) else vars(gk["args"])
        gnet = GraphDeltaValue(d=GA["d_model"], layers=GA["layers"],
                               heads=GA["attn_heads"]).to(args.device)
        gnet.load_state_dict(gk["state_dict"]); gnet.eval()
        PRC = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}

        @torch.no_grad()
        def gdv_score_moves(b, mvs):
            """parent-side single pass; multipv moves matched to move-nodes by (from,to,promo);
            moves beyond the 48-slot cap score -inf (never picked; 2.4 percent of parents)."""
            g, _ = parent_graph(b)
            D = {k: torch.from_numpy(np.asarray(g[k])[None])
                 for k in ("sq", "mv", "mut", "vac", "dsc", "att")}
            D["gl"] = torch.from_numpy(glob_of_board(b)[None])
            D["n_mv"] = torch.tensor([g["n_mv"]], dtype=torch.uint8)
            D["n_att"] = torch.tensor([g["n_att"]], dtype=torch.uint16)
            sqf, mvf, edges, mask = batch_of(D, torch.arange(1), args.device)
            prior, dv = gnet(sqf, mvf, edges, mask)
            n = int(g["n_mv"])
            pr = prior[0, :n].cpu().numpy(); dvn = dv[0, :n].cpu().numpy()
            def zs(x):
                sd = x.std(); return (x - x.mean()) / (sd if sd > 1e-6 else 1.0)
            comb = zs(pr) + args.lam * zs(dvn)
            jmap = {(int(g["mv"][j, 1]), int(g["mv"][j, 2]), int(g["mv"][j, 5])): j
                    for j in range(n)}
            return np.array([comb[jmap[(m.from_square, m.to_square, PRC.get(m.promotion, 0))]]
                             if (m.from_square, m.to_square, PRC.get(m.promotion, 0)) in jmap
                             else -np.inf for m in mvs])

    if args.model == "gdv":
        qk = None                  # scored by gdv_score_moves; no geometry, no child scorer
    elif args.model == "bc":
        # BEHAVIORAL-CLONING PRIOR (Kaveh 2026-08-19): rank children by the policy scorer
        # trained on played-vs-every-sibling. A perfect learner of that objective picks SF's
        # move and is on-track by construction; the number here measures how close training got.
        from catspace.research.components.encoder.approaches.reach_probability.experiments \
            .concept_policy_head import PolicyScorer
        pk = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        PA = pk["args"] if isinstance(pk.get("args"), dict) else vars(pk["args"])
        scorer = PolicyScorer(A["heads"] * 32 + 64 * 32, hidden=PA["hidden"],
                              depth=PA["depth"], arch=PA.get("arch", "mlp"),
                              n_heads=A["heads"], n_codes=A["codes"],
                              n_sq_codes=A["square_codes"]).to(dev)
        scorer.load_state_dict(pk["state_dict"]); scorer.eval()

        @torch.no_grad()
        def score(zc, ig, isq, white_to_move):
            return scorer(zc, ig, isq).cpu().numpy()
        qk = None
    elif args.model == "cef":
        # THE PROPOSED SYSTEM: no geometry at all for value -- score children with the concept
        # model's OWN resolution head, which is the committor we already have. Kaveh 2026-08-18:
        # "we already have a committor head prediction, so we don't even need that."
        out_of0 = np.array([v.split("=")[-1] if "=" in v else v.split(":")[-1] for v in vocab])
        ww = torch.from_numpy((out_of0 == "W").astype(np.float32)).to(dev)
        wb = torch.from_numpy((out_of0 == "L").astype(np.float32)).to(dev)

        @torch.no_grad()
        def score(zc, ig, isq, white_to_move):
            p = torch.softmax(enc.resolution_logits(zc), -1)
            return (p * (ww if white_to_move else wb)).sum(1).cpu().numpy()
        qk = None
    else:
        qk = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    QA = (qk["args"] if isinstance(qk.get("args"), dict) else vars(qk["args"])) if qk else {}
    d_in = A["heads"] * 32 + 64 * 32
    if args.model in ("cef", "bc", "gdv"):
        pass                       # scorer already built above; no geometry to load
    elif args.model == "csg":
        from catspace.research.components.encoder.approaches.reach_probability.experiments \
            .concept_split_geometry import ConceptSplitGeometry
        net = ConceptSplitGeometry(d_in, d_state=QA["d_state"], components=QA["components"],
                                   n_poles=len(vocab), proj_h=QA["proj_h"]).to(dev)
        net.load_state_dict(qk["state_dict"]); net.eval()
        # which poles mean "the side to move WINS"? the resolution vocabulary is white-relative,
        # so it has to be flipped for a black mover.
        out_of = np.array([v.split("=")[-1] if "=" in v else v.split(":")[-1] for v in vocab])
        win_w = torch.from_numpy((out_of == "W").astype(np.float32)).to(dev)
        win_b = torch.from_numpy((out_of == "L").astype(np.float32)).to(dev)

        @torch.no_grad()
        def score(zc, ig, isq, white_to_move):
            """-> higher is better for the ORIGINAL mover."""
            p = net.resolution(net.embed(zc))
            return (p * (win_w if white_to_move else win_b)).sum(1).cpu().numpy()
    else:
        from catspace.research.components.encoder.approaches.reach_probability.experiments \
            .concept_quasimetric import ConceptQuasimetric
        net = ConceptQuasimetric(d_in, d_iqe=QA["d_iqe"], components=QA["components"],
                                 hidden=QA["hidden"], leak_beta=QA["leak_beta"]).to(dev)
        net.load_state_dict(qk["state_dict"]); net.eval()
        gor0 = tr.game_of_row()
        term_rows, _ = tr.terminal_rows()
        rs0 = np.random.default_rng(A["seed"]); perm0 = rs0.permutation(len(tr.start))
        nt0 = max(1, int(0.1 * len(tr.start)))
        tg0 = np.zeros(len(tr.start), bool); tg0[perm0[:nt0]] = True
        keep = ~tg0[gor0[term_rows]]
        rr = np.random.default_rng(0)
        POLE = {}
        for nm, want in (("W", 1), ("B", -1)):
            sel = term_rows[keep & (tr.result[gor0[term_rows]] == want)]
            sel = sel[rr.choice(len(sel), min(args.n_poles, len(sel)), replace=False)]
            POLE[nm] = enc_np(tr.tok[sel], tr.glob[sel])[0]

        @torch.no_grad()
        def score(zc, ig, isq, white_to_move):
            P = POLE["W"] if white_to_move else POLE["B"]
            d = torch.full((len(zc),), float("inf"), device=dev)
            for i in range(len(P)):
                d = torch.minimum(d, net.dist(zc, P[i:i + 1].expand(len(zc), -1)))
            return (-d).cpu().numpy()                 # nearer our pole = better

    gor = tr.game_of_row()
    rs = np.random.default_rng(A["seed"]); perm = rs.permutation(len(tr.start))
    nt = max(1, int(0.1 * len(tr.start)))
    test_g = np.zeros(len(tr.start), bool); test_g[perm[:nt]] = True
    rng = np.random.default_rng(args.seed)
    cand = np.flatnonzero(test_g[gor])
    cand = cand[(cand + 1 < n)]
    rows = cand[rng.choice(len(cand), min(args.n * 6, len(cand)), replace=False)]

    def tok_glob_of(b):
        t = np.zeros(64, np.uint8)
        for q, p in b.piece_map().items():
            t[q] = TOK_PIECES.index((p.piece_type, p.color)) + 1
        g = np.array([int(b.turn), int(b.has_kingside_castling_rights(chess.WHITE)),
                      int(b.has_queenside_castling_rights(chess.WHITE)),
                      int(b.has_kingside_castling_rights(chess.BLACK)),
                      int(b.has_queenside_castling_rights(chess.BLACK)),
                      0 if b.ep_square is None else chess.square_file(b.ep_square) + 1], np.uint8)
        return t, g

    def cat(wp):
        return 2 if wp >= args.win_hi else (1 if wp > args.win_lo else 0)

    VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9,
           chess.KING: 0}
    eng = chess.engine.SimpleEngine.popen_uci(args.engine)
    try:
        eng.configure({"UCI_ShowWDL": True})
    except Exception:
        pass

    ours, rand, mat, best_wp_drop, nsc, ntrack, nlegal = [], [], [], [], 0, [], []
    for r in rows:
        if nsc >= args.n:
            break
        b = board_from_row(tr.tok[r], tr.glob[r])
        if b.is_game_over():
            continue
        mv = list(b.legal_moves)
        if len(mv) < 4:
            continue
        try:
            info = eng.analyse(b, chess.engine.Limit(depth=args.depth), multipv=len(mv))
        except Exception:
            continue
        wp, mvs = {}, []
        for e in info:
            if "pv" not in e or not e["pv"]:
                continue
            m = e["pv"][0]
            s = e["score"].pov(b.turn)
            w = s.wdl().expectation() if s.wdl() is not None else None
            if w is None:
                continue
            wp[m] = w; mvs.append(m)
        if len(mvs) < 4:
            continue
        best = max(wp.values())
        cb = cat(best)
        on = {m: cat(v) >= cb for m, v in wp.items()}
        if all(on.values()) or not any(on.values()):
            continue                                   # no distinction to make here
        nsc += 1
        ntrack.append(np.mean([on[m] for m in mvs])); nlegal.append(len(mvs))

        if gdv_score_moves is not None:
            sc = gdv_score_moves(b, mvs)
        else:
            tks, gbs = [], []
            for m in mvs:
                b.push(m); t, g = tok_glob_of(b); b.pop()
                tks.append(t); gbs.append(g)
            zc, ig, isq = enc_np(np.stack(tks), np.stack(gbs))
            sc = score(zc, ig, isq, bool(tr.glob[r, 0]))
        pick = mvs[int(np.argmax(sc))]
        ours.append(on[pick]); best_wp_drop.append(best - wp[pick])
        rand.append(np.mean([on[m] for m in mvs]))
        mval = [VAL[b.piece_at(m.to_square).piece_type] if b.piece_at(m.to_square) else 0
                for m in mvs]
        mat.append(on[mvs[int(np.argmax(mval))]])
    eng.quit()

    print(f"\n  {nsc} scored positions (held-out games) where the distinction EXISTS")
    print(f"  -- at least one move keeps the result and at least one throws it away")
    print(f"  mean legal moves {np.mean(nlegal):.1f}, of which "
          f"{100*np.mean(ntrack):.1f}% are on-track\n")
    print(f"  {'method':<26} {'on winning track':>18}")
    print(f"  {'OUR FIELD (' + args.model + ')':<26} {100*np.mean(ours):>17.1f}%")
    print(f"  {'random legal move':<26} {100*np.mean(rand):>17.1f}%")
    print(f"  {'material-greedy':<26} {100*np.mean(mat):>17.1f}%")
    print(f"\n  when our pick is OFF-track, mean win-probability given up: "
          f"{np.mean([d for d, o in zip(best_wp_drop, ours) if not o]):.3f}")
    print(f"  mean win-probability given up over ALL picks: {np.mean(best_wp_drop):.3f}")


if __name__ == "__main__":
    main()
