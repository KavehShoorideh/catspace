"""Future-KIND labels: which sort of future does each move open?

Stage 1 of the design in docs/FUTURE_KINDS.md. A subgoal is a cluster in RESIDUAL
future-concept-occupancy space, so building the labels has three stages with very
different costs, and they are deliberately separable:

  pv     Stockfish multipv gives, for each of the top moves, its principal variation
         -- a sample of that move's future under best play. CPU only, ~42 ms/position
         at depth 10, and the output is just UCI text, so it can run beside a live
         trainer without touching the GPU or the disk budget.
  fit    encode PV positions with a FROZEN reference field, form each move's
         discounted concept occupancy, subtract the position's mean (leaving what the
         MOVE changes), and fit K centroids ONCE. Needs the GPU.
  assign nearest-centroid for every move, plus the per-position value spread that
         weights the loss. Needs the GPU.

Splitting `pv` out is what lets the expensive half wait for the GPU to be free.

Why the residual: measured (JOURNAL 2026-08-24), clustering the ABSOLUTE occupancy
produces a position recogniser -- only 17% of positions reach more than one kind,
because every line starts from the same place. The residual gives kinds worth 1.74
pawns at every position.
"""
import argparse
import glob
import os
import time
import numpy as np

PVMAX = 10          # plies of continuation kept per move
NMV = 12            # moves labelled per position: the beam that matters (beam 8 already
                    # retains a winning move 94.7% of the time)


def _mv_triplet(m):
    return (m.from_square, m.to_square, (m.promotion or 0))


def pv_shard(shard, out, depth, sf_path):
    """One shard -> its PV file. Resumable: an existing output is left alone."""
    import chess, chess.engine
    d = np.load(shard)
    fens = [str(x) for x in d["fen"]]
    n = len(fens)
    rmv = np.full((n, NMV, 3), 255, np.uint8)
    pv = np.full((n, NMV, PVMAX, 3), 255, np.uint8)
    pvn = np.zeros((n, NMV), np.uint8)
    val = np.zeros((n, NMV), np.float16)

    eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    eng.configure({"Threads": 1, "Hash": 32})
    for i, f in enumerate(fens):
        try:
            b = chess.Board(f)
        except Exception:
            continue
        if b.is_game_over():
            continue
        nl = b.legal_moves.count()
        if nl == 0:
            continue
        try:
            info = eng.analyse(b, chess.engine.Limit(depth=depth),
                               multipv=min(NMV, nl))
        except Exception:
            try:
                eng.quit()
            except Exception:
                pass
            eng = chess.engine.SimpleEngine.popen_uci(sf_path)
            eng.configure({"Threads": 1, "Hash": 32})
            continue
        for j, e in enumerate(info[:NMV]):
            line = e.get("pv") or []
            if not line:
                continue
            rmv[i, j] = _mv_triplet(line[0])
            sc = e.get("score")
            # POV of the side to move, so a move's value means the same thing in every
            # row regardless of colour
            val[i, j] = (sc.pov(b.turn).score(mate_score=10000) / 100.0) if sc else 0.0
            k = 0
            for m in line[1:1 + PVMAX]:
                pv[i, j, k] = _mv_triplet(m)
                k += 1
            pvn[i, j] = k
    eng.quit()
    np.savez_compressed(out, rmv=rmv, pv=pv, pvn=pvn, val=val)
    return int((pvn > 0).sum())


def _worker(a):
    shard, out, depth, sf = a
    if os.path.exists(out):
        return (out, -1, 0.0)
    t0 = time.time()
    try:
        k = pv_shard(shard, out + ".tmp.npz", depth, sf)
    except Exception as e:
        return (out, -2, str(e))
    os.replace(out + ".tmp.npz", out)
    return (out, k, time.time() - t0)



# ---------------------------------------------------------------------------
# stages fit / assign -- these need the frozen reference field and the GPU
# ---------------------------------------------------------------------------

def _graph_of(fen):
    """Graph arrays for one FEN. Module-level so a Pool can pickle it."""
    import chess
    from catspace.research.tools.training_infra.gen_move_graphs import (
        parent_graph, glob_of_board)
    b = chess.Board(fen)
    g, _ = parent_graph(b)
    out = {k: np.asarray(g[k]) for k in
           ("sq", "mv", "mut", "att", "n_mv", "n_att", "vac", "dsc", "pslot")}
    out["gl"] = glob_of_board(b)
    return out


def encode_boards(ke, boards, bs=256, dev=None, jobs=1, pool=None):
    """Batched concept encode: (B,) boards -> (B, d_state).

    Mirrors the TRAINER's batch_of(), not the engine's single-board _encode():
    unbatched encoding costs ~100 ms/position, which turns a labelling pass into
    days. A gate below asserts the two agree, because a silently different batched
    path would poison every label without failing anything.
    """
    import torch
    from catspace.research.tools.training_infra.gen_move_graphs import (
        parent_graph, glob_of_board)
    from catspace.research.components.encoder.approaches.reach_probability         .experiments.graph_move_policy import featurize, MV_CAP
    from catspace.research.components.encoder.approaches.reach_probability.src         .trajectories import normalise_elo
    dev = dev or ke.dev
    out = []
    # graph construction is 0.95 ms/position and single-threaded it DOMINATES the pass
    # (2.3 core-hours for 50 shards, against ~10 min of GPU). It parallelises cleanly
    # because every board is known up front.
    fens = [b.fen() for b in boards]
    if pool is not None:
        graphs = pool.map(_graph_of, fens, chunksize=64)
    else:
        graphs = [_graph_of(f) for f in fens]
    for s0 in range(0, len(boards), bs):
        gs = graphs[s0:s0 + bs]
        B = len(gs)
        bt = {k: torch.from_numpy(np.stack([g[k] for g in gs])).to(dev)
              for k in ("sq", "mv", "mut", "att", "n_mv", "n_att")}
        bt["gl"] = torch.from_numpy(np.stack([g["gl"] for g in gs])).to(dev)
        psl = (torch.from_numpy(np.stack([g["pslot"] for g in gs]))
               .long().to(dev) if ke.gen5 else None)
        sqf, mvf, edges, mask = featurize(bt, dev, n_nodes=ke.field.n_nodes)
        off = (torch.arange(B, device=dev) * ke.field.n_nodes)[:, None]
        jj = torch.arange(MV_CAP, device=dev)[None, :]
        for nm in ("vac", "dsc"):
            a2 = torch.from_numpy(np.stack([g[nm] for g in gs])).long().to(dev)
            val = (a2 != 255) & mask[:, :, None]
            edges[nm] = (((jj + 64 + off)[:, :, None].expand_as(a2))[val],
                         (a2 + off[:, :, None])[val])
        # fixed rating for the reference encode: a label must not depend on who is
        # sitting opposite, only on what the position becomes
        el = torch.from_numpy(normalise_elo(
            np.full((B, 2), 2000.0, np.float32))).to(dev)
        with torch.no_grad():
            zq = ke.field.encode(sqf, mvf, edges, mask, elo=el, pslot=psl)[0]
        out.append(zq[:, :ke.field.d_state].float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, ke.field.d_state), np.float32)


def shard_occupancy(ke, shard, fkfile, gamma=0.9, bs=256, dev=None,
                    max_rows=0, pool=None):
    """One shard -> (residual occupancy per labelled move, row index, move triplet,
    value). The residual is what the MOVE changes: the position's mean future is
    subtracted, because clustering the absolute occupancy yields a position
    recogniser (measured: only 17% of positions reach >1 kind)."""
    import chess
    d, p = np.load(shard), np.load(fkfile)
    N = len(d["fen"]) if not max_rows else min(max_rows, len(d["fen"]))
    boards, owner = [], []          # owner = (row, move_slot, discount weight)
    for i in range(N):
        try:
            b0 = chess.Board(str(d["fen"][i]))
        except Exception:
            continue
        for j in range(NMV):
            n = int(p["pvn"][i, j])
            if n == 0:
                continue
            b = b0.copy()
            f_, t_, pr = (int(x) for x in p["rmv"][i, j])
            try:
                b.push(chess.Move(f_, t_, promotion=pr or None))
            except Exception:
                continue
            for k in range(n):
                f2, t2, p2 = (int(x) for x in p["pv"][i, j, k])
                try:
                    b.push(chess.Move(f2, t2, promotion=p2 or None))
                except Exception:
                    break
                boards.append(b.copy())
                owner.append((i, j, gamma ** k))
    if not boards:
        return None
    phi = encode_boards(ke, boards, bs=bs, dev=dev, pool=pool)
    acc = {}
    for (i, j, w), v in zip(owner, phi):
        a = acc.get((i, j))
        if a is None:
            acc[(i, j)] = [v * w, w]
        else:
            a[0] += v * w
            a[1] += w
    rows = sorted(acc)
    M = np.stack([acc[k][0] / max(acc[k][1], 1e-9) for k in rows]).astype(np.float32)
    ri = np.array([k[0] for k in rows], np.int32)
    mj = np.array([k[1] for k in rows], np.int8)
    # residual: subtract each POSITION's mean future
    R = M.copy()
    for i in np.unique(ri):
        sel = ri == i
        if sel.sum() > 1:
            R[sel] -= R[sel].mean(0)
    return R, ri, mj, p


def _engine(ckpt, dev):
    from catspace.research.components.planner.approaches.quasimetric_nav \
        .koopman_search import KoopmanEngine
    ke = KoopmanEngine(ckpt)
    if dev:
        import torch
        ke.field.to(dev)
        ke.dev = torch.device(dev)
    return ke


def _pool(a):
    if a.jobs <= 1:
        return None
    from multiprocessing import Pool
    return Pool(a.jobs)


def stage_fit(a):
    """Fit the K centroids ONCE, on a sample, and freeze them.

    Frozen because a clustering that moves with the encoder is a moving target --
    the bootstrap spiral the loss-interaction review caught once already.
    """
    ke = _engine(a.ref_ckpt, a.device)
    pool = _pool(a)
    sh = sorted(glob.glob(os.path.join(a.shards, "*.npz")))
    have = [s for s in sh if os.path.exists(os.path.join(a.out, os.path.basename(s)))]
    use = have[:a.fit_shards]
    print(f"[fit] {len(use)} shards, {a.rows_per_shard} rows each, K={a.K}", flush=True)
    Rs = []
    for s in use:
        r = shard_occupancy(ke, s, os.path.join(a.out, os.path.basename(s)),
                            gamma=a.gamma, bs=a.batch, dev=a.device,
                            max_rows=a.rows_per_shard, pool=pool)
        if r is not None:
            Rs.append(r[0])
            print(f"[fit]   {os.path.basename(s)} -> {len(r[0])} moves", flush=True)
    R = np.concatenate(Rs)
    mu, sd = R.mean(0), R.std(0) + 1e-6
    Zc = (R - mu) / sd
    from sklearn.cluster import KMeans
    km = KMeans(a.K, n_init=10, random_state=0).fit(Zc)
    cnt = np.bincount(km.labels_, minlength=a.K)
    np.savez(os.path.join(a.out, "centroids.npz"), mu=mu, sd=sd,
             C=km.cluster_centers_.astype(np.float32), K=a.K, gamma=a.gamma,
             ref=os.path.basename(a.ref_ckpt))
    print(f"[fit] {len(R):,} moves -> {a.K} kinds | occupancy {cnt.min()}-{cnt.max()} "
          f"(uniform would be {len(R)//a.K}) | saved centroids.npz", flush=True)


def stage_assign(a):
    """Nearest frozen centroid for every labelled move, plus the per-position value
    spread that weights the loss."""
    ke = _engine(a.ref_ckpt, a.device)
    pool = _pool(a)
    C = np.load(os.path.join(a.out, "centroids.npz"))
    mu, sd, cen, K = C["mu"], C["sd"], C["C"], int(C["K"])
    sh = sorted(glob.glob(os.path.join(a.shards, "*.npz")))
    done = 0
    for s in sh:
        f = os.path.join(a.out, os.path.basename(s))
        if not os.path.exists(f):
            continue
        if "fkind48" in np.load(f).files and not a.force:
            done += 1
            continue
        if a.assign_shards and done >= a.assign_shards:
            break
        r = shard_occupancy(ke, s, f, gamma=a.gamma, bs=a.batch,
                            dev=a.device, max_rows=a.assign_rows, pool=pool)
        if r is None:
            continue
        R, ri, mj, p = r
        Zc = (R - mu) / sd
        d2 = ((Zc[:, None, :] - cen[None]) ** 2).sum(-1)
        kid = d2.argmin(1).astype(np.int8)
        n = len(p["pvn"])
        fkind = np.full((n, NMV), -1, np.int8)
        fkind[ri, mj] = kid
        # weight = what the CHOICE is worth here: spread of teacher value across the
        # kinds actually reachable. Mate scores are clipped -- decisive is decisive,
        # and unclipped they are +-100 pawns and would swamp every other position.
        val = np.clip(p["val"].astype(np.float32), -10.0, 10.0)
        spread = np.zeros(n, np.float32)
        for i in range(n):
            sel = fkind[i] >= 0
            if sel.sum() < 2:
                continue
            ks, vs = fkind[i][sel], val[i][sel]
            means = [vs[ks == u].mean() for u in np.unique(ks)]
            spread[i] = (max(means) - min(means)) if len(means) > 1 else 0.0
        # REMAP to the shard's own 48-move order, offline. The trainer indexes moves
        # by shard slot; doing this join at train time would put a silent alignment
        # bug on the hot path, which is how the lift came to slice the board at the
        # wrong offsets for a whole run.
        MV48 = d["mv"].shape[1]
        fk48 = np.full((n, MV48), -1, np.int8)
        miss = 0
        for i in range(n):
            k = int(d["n_mv"][i])
            if k == 0:
                continue
            exact, byft = {}, {}
            for c in range(k):
                key = (int(d["mv"][i, c, 1]), int(d["mv"][i, c, 2]),
                       int(d["mv"][i, c, 5]))
                exact[key] = c
                byft.setdefault(key[:2], []).append(c)
            for j in range(NMV):
                if fkind[i, j] < 0:
                    continue
                f_, t_, pr = (int(x) for x in p["rmv"][i, j])
                c = exact.get((f_, t_, pr))
                if c is None:
                    cand = byft.get((f_, t_), [])
                    c = cand[0] if len(cand) == 1 else None
                if c is None:
                    miss += 1
                    continue
                fk48[i, c] = fkind[i, j]
        out = {k: p[k] for k in p.files}
        out["fkind"] = fkind
        out["fkind48"] = fk48
        out["fspread"] = spread.astype(np.float16)
        np.savez_compressed(f + ".tmp.npz", **out)
        os.replace(f + ".tmp.npz", f)
        done += 1
        print(f"[assign] {os.path.basename(s)}: {int((fk48>=0).sum())} moves "
              f"mapped ({miss} unmapped), "
              f"{100*np.mean(spread>0.25):.0f}% of rows have a choice worth "
              f">0.25 pawns (mean spread {spread[spread>0].mean():.2f})", flush=True)
    print(f"[assign] {done} shards carry future-kind labels", flush=True)


def stage_gate(a):
    """Assert the batched encode IS the single-board encode.

    Every label depends on this. A batched path that quietly differs would poison
    all of them without failing anything -- the exact silent-failure shape that
    docs/PROMPTS.md exists for. Two separate claims:
      * at batch 1 it must be EXACTLY equal (structural identity of the code path);
      * batched, it may differ only in a negligible fraction of near-zero cells
        (measured: 6 of 25,280, all in the piece/hist tail at magnitude 0.026 against
        a state whose mean is 0.828 -- transformer padding kernels, not our code).
    """
    import chess
    ke = _engine(a.ref_ckpt, a.device)
    d = np.load(sorted(glob.glob(os.path.join(a.shards, "*.npz")))[0])
    boards = [chess.Board(str(d["fen"][i])) for i in range(12)]
    boards = [b for b in boards if not b.is_game_over()]
    ref = []
    for b in boards:
        ke._encode(b)
        ref.append(ke._last[0][0, :ke.field.d_state].cpu().numpy())
    ref = np.stack(ref)
    e1 = np.abs(encode_boards(ke, boards, bs=1, dev=a.device) - ref).max()
    assert e1 == 0.0, f"batch-1 path is NOT the single-board path: {e1}"
    dif = np.abs(encode_boards(ke, boards, bs=8, dev=a.device) - ref)
    frac = float((dif > 1e-4).mean())
    assert frac < 1e-3, f"batching changes {100*frac:.3f}% of cells -- too many"
    print(f"[gate] batch-1 EXACT; batch-8 differs in {100*frac:.3f}% of cells "
          f"(max {dif.max():.1e}) -- PASS", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="pv",
                    choices=("pv", "fit", "assign", "gate"))
    ap.add_argument("--shards", default="data/derived/gko1_shards")
    ap.add_argument("--out", default="data/derived/gko1_fkind")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="0 = every shard")
    ap.add_argument("--sf", default="/opt/homebrew/bin/stockfish")
    ap.add_argument("--ref-ckpt", default="")
    ap.add_argument("--device", default="")
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--fit-shards", type=int, default=6)
    ap.add_argument("--rows-per-shard", type=int, default=400)
    ap.add_argument("--assign-shards", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=1,
                    help="processes for graph construction (the pass's "
                         "real bottleneck; the GPU is not)")
    ap.add_argument("--assign-rows", type=int, default=0,
                    help="0 = every row; smaller makes a quick test set")
    ap.add_argument("--force", action="store_true",
                    help="re-assign shards that already carry labels")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    if a.stage == "gate":
        return stage_gate(a)
    if a.stage == "fit":
        return stage_fit(a)
    if a.stage == "assign":
        return stage_assign(a)
    sh = sorted(glob.glob(os.path.join(a.shards, "*.npz")))
    if a.limit:
        sh = sh[:a.limit]
    jobs = [(s, os.path.join(a.out, os.path.basename(s)), a.depth, a.sf) for s in sh]
    todo = [j for j in jobs if not os.path.exists(j[1])]
    print(f"[fkind] {len(sh)} shards, {len(todo)} to do, {a.workers} workers, "
          f"depth {a.depth}, top-{NMV} moves, {PVMAX}-ply lines", flush=True)

    from multiprocessing import Pool
    t0, done, tot = time.time(), 0, 0
    with Pool(a.workers) as p:
        for out, k, dt in p.imap_unordered(_worker, todo):
            done += 1
            if k == -2:
                print(f"[fkind] FAILED {os.path.basename(out)}: {dt}", flush=True)
                continue
            tot += max(k, 0)
            el = time.time() - t0
            eta = el / max(done, 1) * (len(todo) - done)
            print(f"[fkind] {done}/{len(todo)} {os.path.basename(out)} "
                  f"{k} lines in {dt:.0f}s | elapsed {el/60:.1f}m ETA {eta/60:.1f}m",
                  flush=True)
    print(f"[fkind] done: {tot:,} PV lines over {done} shards in "
          f"{(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
