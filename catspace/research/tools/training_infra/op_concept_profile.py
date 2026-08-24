"""Where does a move's operator delta LAND across concepts?

Kaveh 2026-08-24: "a single move is not going to change the whole board, so we do
expect some things to be the same. But on a per concept level, one concept ought to
change a lot, and others not so much. It's not like every concept should be near
identity -- one concept should not be."

That is a distributional claim and it is testable. For K(m)z - z we group the state
dimensions into their semantic units and ask how the delta's mass is spread:

    board concepts   g_heads x d_code   (the 64 global VQ concept heads)
    squares          64 x d_sq          (one unit per board square)
    piece slots      32 x d_piece       (one unit per canonical piece)

A HEALTHY operator concentrates: a few units carry most of the delta and the rest
sit near identity. A DEGENERATE one spreads the same small change over everything,
which is what "99pct move-independent" looks like from the inside.

Reported per grouping:
  * top-1 / top-5 share of total delta mass
  * participation ratio -- the effective NUMBER of units moved (1 = perfectly
    concentrated on one unit, N = perfectly uniform). This is the headline.
  * uniform baseline for reference.

Plus an interpretability check that needs no concept labels: after moving a piece,
is THAT piece's slot the one that changes most? If the operator has learned
anything structural, the moved piece should rank top by its own delta.
"""
import argparse
import numpy as np
import torch


def unit_norms(delta, start, n_units, width):
    """L2 norm of the delta within each semantic unit -> (B, n_units)."""
    blk = delta[:, start:start + n_units * width]
    return blk.reshape(len(delta), n_units, width).norm(dim=-1)


def spread(x):
    """Concentration of a non-negative (B, N) row: mean top-1 share, top-5 share,
    and participation ratio = (sum x)^2 / sum(x^2), the effective count of units
    carrying the mass."""
    p = x / x.sum(-1, keepdim=True).clamp(min=1e-12)
    srt = p.sort(-1, descending=True).values
    pr = (x.sum(-1) ** 2 / (x ** 2).sum(-1).clamp(min=1e-12))
    return (float(srt[:, 0].mean()), float(srt[:, :5].sum(-1).mean()),
            float(pr.mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/gko4b_step4000.pt")
    ap.add_argument("--holdout", default="data/derived/gko1_holdout_sf")
    ap.add_argument("--n_pos", type=int, default=24)
    ap.add_argument("--codes", action="store_true",
                    help="DISCRETE profile: how many VQ CODES flip per move. The "
                         "concepts are quantised, so a diffuse continuous delta is "
                         "compatible with sparse code changes -- and sparsity at the "
                         "code level is the thing planning would actually use.")
    ap.add_argument("--truth", action="store_true",
                    help="profile the TRUE delta (encode(child) - encode(parent)) "
                         "instead of the operator's. Decides whether a diffuse "
                         "operator delta is the operator's failure or a faithful "
                         "copy of a representation in which every move is global.")
    args = ap.parse_args()

    import glob, os, chess
    from catspace.research.components.planner.approaches.quasimetric_nav \
        .koopman_search import KoopmanEngine
    ke = KoopmanEngine(args.ckpt, reencode_budget=12)
    if not ke.gen5:
        raise SystemExit("per-concept profile targets gen5 (explicit A + piece slots)")

    # gen5 state layout: [board 64x8 | squares 64x16 | pieces 32x4 | hist 128]
    A = dict(torch.load(args.ckpt, map_location="cpu",
                        weights_only=False)["args"])
    n_head, d_code, d_sq, n_piece, d_piece = A["g_heads"], A["d_code"], 16, 32, 4
    b0 = 0
    s0 = b0 + n_head * d_code
    p0 = s0 + 64 * d_sq
    print(f"[prof] {args.ckpt.split('/')[-1]} | board {n_head}x{d_code} @ {b0} | "
          f"squares 64x{d_sq} @ {s0} | pieces {n_piece}x{d_piece} @ {p0}")

    fens = []
    for f in sorted(glob.glob(os.path.join(args.holdout, "*.npz"))):
        d = np.load(f)
        fens += [x.decode() if isinstance(x, bytes) else str(x) for x in d["fen"]]
        if len(fens) >= args.n_pos * 3:
            break

    if args.codes:
        import chess as _ch
        from catspace.research.components.encoder.approaches.reach_probability \
            .experiments.graph_move_policy import featurize as _ftz
        flips = {"board": [], "square": [], "piece": []}
        n_mv_tot = 0
        for fen in fens:
            if n_mv_tot >= args.n_pos * 20:
                break
            try:
                bd = _ch.Board(fen)
            except Exception:
                continue
            if bd.is_game_over() or len(list(bd.legal_moves)) < 4:
                continue

            def codes_of(board):
                """discrete concept codes: board heads, per-square, per-piece."""
                from catspace.research.tools.training_infra.gen_move_graphs import (
                    parent_graph, glob_of_board, piece_slots)
                from catspace.research.components.encoder.approaches \
                    .reach_probability.src.trajectories import normalise_elo
                g, _ = parent_graph(board)
                b = {k: torch.from_numpy(np.asarray(g[k])[None]) for k in
                     ("sq", "mv", "mut", "att", "n_mv", "n_att")}
                b["gl"] = torch.from_numpy(glob_of_board(board)[None])
                sqf, mvf, edges, mask = _ftz(b, ke.dev, n_nodes=ke.field.n_nodes)
                off = torch.zeros(1, 1, dtype=torch.long)
                jj = torch.arange(mvf.shape[1])[None, :]
                for nm in ("vac", "dsc"):
                    a2 = torch.from_numpy(np.asarray(g[nm])[None]).long()
                    val = (a2 != 255) & mask[:, :, None]
                    edges[nm] = (((jj + 64 + off)[:, :, None].expand_as(a2))[val],
                                 (a2 + off[:, :, None])[val])
                el = torch.from_numpy(normalise_elo(
                    np.full((1, 2), 2000.0, np.float32)))
                psl = torch.from_numpy(np.asarray(g["pslot"])[None]).long()
                with torch.no_grad():
                    out = ke.field.encode(sqf, mvf, edges, mask, elo=el, pslot=psl)
                return out[1]     # g_ids: the discrete concept code indices

            c0 = codes_of(bd)
            if c0 is None:
                raise SystemExit("field.encode did not return code ids at [1]")
            c0 = c0.reshape(-1)
            for mv in list(bd.legal_moves)[:12]:
                bd.push(mv)
                c1 = codes_of(bd).reshape(-1)
                bd.pop()
                n = min(len(c0), len(c1))
                flips["board"].append(float((c0[:n] != c1[:n]).float().mean()))
                n_mv_tot += 1
        f = np.array(flips["board"])
        print(f"\n  DISCRETE code flips per move ({n_mv_tot} moves, {len(c0)} codes):")
        print(f"    mean {100*f.mean():.1f}% of codes change | median {100*np.median(f):.1f}%"
              f" | min {100*f.min():.1f}% max {100*f.max():.1f}%")
        print(f"    => effective codes changed: {f.mean()*len(c0):.1f} of {len(c0)}")
        raise SystemExit(0)

    agg = {k: [] for k in ("board", "square", "piece")}
    mover_rank, mover_n = [], 0
    for fen in fens:
        if len(agg["board"]) >= args.n_pos:
            break
        try:
            bd = chess.Board(fen)
        except Exception:
            continue
        if bd.is_game_over() or len(list(bd.legal_moves)) < 4:
            continue
        za, cond, moves, _ = ke._encode(bd)
        with torch.no_grad():
            zin = za.expand(len(cond), -1)
            if args.truth:
                # the REAL next state: encode each child board directly
                rows = []
                for mv in moves:
                    bd.push(mv)
                    rows.append(ke._encode(bd)[0][0])
                    bd.pop()
                zc = torch.stack(rows)
            else:
                zc = ke.ops(zin, cond[:, 0], cond[:, 1], cond[:, 2])
            delta = (zc - zin).cpu()
        agg["board"].append(unit_norms(delta, b0, n_head, d_code))
        agg["square"].append(unit_norms(delta, s0, 64, d_sq))
        pn = unit_norms(delta, p0, n_piece, d_piece)
        agg["piece"].append(pn)

        # does the MOVED piece's slot move most? (needs no concept labels)
        from catspace.research.tools.training_infra.gen_move_graphs import piece_slots
        psl = piece_slots(bd)
        for i, mv in enumerate(moves):
            slot = int(psl[mv.from_square])
            if slot == 255:
                continue
            order = pn[i].argsort(descending=True)
            rank = int((order == slot).nonzero()[0, 0])
            mover_rank.append(rank)
            mover_n += 1

    print(f"[prof] {len(agg['board'])} positions, "
          f"{sum(len(a) for a in agg['board'])} moves\n")
    print("  grouping   units   top-1 share   top-5 share   EFFECTIVE UNITS MOVED")
    for name, n_units in (("board", n_head), ("square", 64), ("piece", n_piece)):
        x = torch.cat(agg[name])
        t1, t5, pr = spread(x)
        print(f"  {name:<9} {n_units:>4}      {t1:6.1%}       {t5:6.1%}        "
              f"{pr:6.1f}  (uniform would be {n_units})")
    print()
    if mover_n:
        r = np.array(mover_rank)
        print(f"  moved-piece check ({mover_n} moves): the moving piece's own slot "
              f"ranks #{np.median(r) + 1:.0f} of {n_piece} by delta (median)")
        print(f"    top-1 {100 * (r == 0).mean():.1f}%  top-3 {100 * (r < 3).mean():.1f}%"
              f"  |  chance would be {100 / n_piece:.1f}% / {300 / n_piece:.1f}%")


if __name__ == "__main__":
    main()
