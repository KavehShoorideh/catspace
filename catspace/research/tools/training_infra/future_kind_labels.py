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
    # Build graphs PER BATCH, not all at once. Materialising every graph up front is
    # fine at 480 rows/shard (28k positions) and thrashes at full rows (121k positions,
    # ~530 MB of graph arrays plus pickling), which showed up as the main process
    # sitting at 3.6% CPU waiting on the pool while RSS climbed past 2 GB.
    # `boards` are already FENs (strings) -- the pool needs strings anyway, and
    # pickling Board objects to workers is slower than rebuilding from FEN
    fens = [b if isinstance(b, str) else b.fen() for b in boards]
    _tg = [0.0]
    for s0 in range(0, len(boards), bs):
        import time as _tt
        _g0 = _tt.time()
        chunk = fens[s0:s0 + bs]
        gs = (pool.map(_graph_of, chunk, chunksize=32) if pool is not None
              else [_graph_of(f) for f in chunk])
        _tg[0] += _tt.time() - _g0
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
    print(f"[enc] graph-build {_tg[0]:.0f}s of the pass", flush=True)
    return np.concatenate(out) if out else np.zeros((0, ke.field.d_state), np.float32)


def _plies(n, k):
    """Stratified subsample of a k-of-n plies, always including the LAST.

    The occupancy is a discounted AVERAGE over the line, so a stratified subsample
    estimates the same quantity at a fraction of the cost -- and cost is the binding
    constraint here (encoding every ply of every line is 6.6 GPU-hours for 50 shards).
    Subsampling PLIES is strictly better than shortening the HORIZON, which would
    change what the label means: the queens-off signal that motivated all this was
    measured over ~8-10 plies.
    """
    if k <= 0 or n <= k:
        return list(range(n))
    return sorted({min(n - 1, int(round(i * (n - 1) / (k - 1)))) for i in range(k)})


def shard_occupancy(ke, shard, fkfile, gamma=0.9, bs=256, dev=None,
                    max_rows=0, pool=None, max_moves=0, pv_samples=0):
    """One shard -> (residual occupancy per labelled move, row index, move triplet,
    value). The residual is what the MOVE changes: the position's mean future is
    subtracted, because clustering the absolute occupancy yields a position
    recogniser (measured: only 17% of positions reach >1 kind)."""
    import chess
    d, p = np.load(shard), np.load(fkfile)
    pv_a, pvn_a, rmv_a = p["pv"], p["pvn"], p["rmv"]        # decompress ONCE
    N = len(d["fen"]) if not max_rows else min(max_rows, len(d["fen"]))
    import time as _t
    _tb = _t.time()
    boards, owner = [], []          # owner = (row, move_slot, discount weight)
    for i in range(N):
        try:
            b0 = chess.Board(str(d["fen"][i]))
        except Exception:
            continue
        for j in range(max_moves or NMV):
            n = int(pvn_a[i, j])
            if n == 0:
                continue
            b = b0.copy()
            f_, t_, pr = (int(x) for x in rmv_a[i, j])
            try:
                b.push(chess.Move(f_, t_, promotion=pr or None))
            except Exception:
                continue
            keep = set(_plies(n, pv_samples))
            for k in range(n):
                f2, t2, p2 = (int(x) for x in pv_a[i, j, k])
                try:
                    b.push(chess.Move(f2, t2, promotion=p2 or None))
                except Exception:
                    break
                if k not in keep:
                    continue          # walked (board state must advance) but not encoded
                boards.append(b.fen())
                owner.append((i, j, gamma ** k))
    if not boards:
        return None
    _bt = _t.time() - _tb
    _t0 = _t.time()
    phi = encode_boards(ke, boards, bs=bs, dev=dev, pool=pool)
    _dt = _t.time() - _t0
    _t1 = _t.time()
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
    print(f"[occ] {os.path.basename(shard)}: {len(boards)} pos | board-build {_bt:.0f}s "
          f"| encode {_dt:.0f}s ({_dt/max(len(boards),1)*1000:.2f} ms/pos) | "
          f"accumulate {_t.time()-_t1:.0f}s", flush=True)
    return R, ri, mj, p


def _engine(ckpt, dev):
    from catspace.research.components.planner.approaches.quasimetric_nav \
        .koopman_search import KoopmanEngine
    ke = KoopmanEngine(ckpt)
    if dev:
        import torch
        import torch.nn as nn
        # move EVERY module, not just the field: _encode also runs the lift, and
        # leaving it on CPU is the engine's known device-mismatch failure. This
        # crashed the gate on mps before it could crash the unattended pass.
        for name in dir(ke):
            m = getattr(ke, name, None)
            if isinstance(m, nn.Module):
                m.to(dev)
        ke.dev = torch.device(dev)
    return ke


def sinkhorn_assign(D2, eps=0.05, iters=30):
    """Balanced assignment of moves to kinds (Sinkhorn-Knopp equipartition).

    Plain nearest-centroid gave one kind 48% of all moves and left the effective
    number of kinds at 4.8 of 6. That is the classic degenerate clustering assignment,
    and the standard fix is an EQUIPARTITION constraint solved with Sinkhorn-Knopp --
    Asano et al. (Self-Labelling, ICLR 2020) and SwAV (Caron et al., NeurIPS 2020),
    where it is what prevents every sample collapsing onto one prototype.

    It matters here beyond tidiness: this design leans on log K being the price of
    declining to predict, and log K is only an honest ceiling if all K kinds carry
    real mass. With one kind at 48% the true ceiling is much lower than advertised.

    Balance is enforced GLOBALLY (over all moves in the shard), never within a
    position -- forcing one position's moves to spread across kinds would invent
    distinctions where a position genuinely offers only one kind of future.

    D2: (N, K) squared distances to centroids. Returns (N,) int labels.
    """
    Q = np.exp(-D2 / max(eps, 1e-6) / (D2.std() + 1e-9))
    Q = np.clip(Q, 1e-30, None)
    Q /= Q.sum()
    N, K = Q.shape
    for _ in range(iters):
        Q /= Q.sum(0, keepdims=True).clip(1e-30)      # each kind equally used
        Q /= K
        Q /= Q.sum(1, keepdims=True).clip(1e-30)      # each move assigned once
        Q /= N
    return Q.argmax(1).astype(np.int8)


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
    rc = os.path.join(a.out, "_residuals.npy")
    if a.refit and os.path.exists(rc):
        return fit_from_residuals(np.load(rc), a)
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
                            max_rows=a.rows_per_shard, pool=pool,
                            max_moves=a.max_moves, pv_samples=a.pv_samples)
        if r is not None:
            Rs.append(r[0])
            print(f"[fit]   {os.path.basename(s)} -> {len(r[0])} moves", flush=True)
    R = np.concatenate(Rs)
    np.save(os.path.join(a.out, "_residuals.npy"), R)   # refits become instant
    fit_from_residuals(R, a)


def fit_from_residuals(R, a):
    """Standardise -> PCA -> K-means, and report whether the kinds are usable.

    PCA first because K-means in 2528 dims on a few thousand samples suffers distance
    concentration: the first attempt (no PCA) produced occupancy 3-1155 where uniform
    is 432, so one 'kind' swallowed 22% of moves and another had 3 members. Rare
    classes are not subgoals, they are noise the head can never learn, and they make
    the effective K far smaller than the nominal one -- which matters because log K is
    the loss ceiling the whole design leans on.
    """
    from sklearn.cluster import KMeans
    mu, sd = R.mean(0), R.std(0) + 1e-6
    Zc = (R - mu) / sd
    Zc -= Zc.mean(0)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    npc = min(a.n_pca, Vt.shape[0])
    W = Vt[:npc].T.astype(np.float32)
    ev = float((S[:npc] ** 2).sum() / (S ** 2).sum())
    P = Zc @ W
    # UNIT-NORMALISE, then cluster directions. Successor Options (Ramesh et al.,
    # IJCAI 2019) -- the direct precedent for "subgoal = cluster in successor-feature
    # space" -- selects sub-goals by COSINE similarity to the cluster centre, and the
    # reason applies here: which KIND of future a move opens is a direction of concept
    # change; how far it goes is a magnitude, and mixing the two is what parks a big
    # cluster at the origin. Measured on real residuals (686 moves), against euclidean:
    # kinds/row 4.63 vs 4.49, within-row value gap 2.47 vs 2.37 pawns.
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    km = KMeans(a.K, n_init=20, random_state=0).fit(P)
    cnt = np.bincount(km.labels_, minlength=a.K)
    # effective number of kinds actually used: 1 = one kind swallows everything,
    # K = perfectly balanced. This is the number that decides if K is honest.
    pr = float(cnt.sum() ** 2 / max((cnt.astype(float) ** 2).sum(), 1e-9))
    np.savez(os.path.join(a.out, "centroids.npz"), mu=mu, sd=sd, W=W,
             C=km.cluster_centers_.astype(np.float32), K=a.K, gamma=a.gamma,
             ref=os.path.basename(a.ref_ckpt))
    print(f"[fit] {len(R):,} moves | PCA {npc} dims keep {100*ev:.0f}% of residual "
          f"variance | {a.K} kinds, occupancy {cnt.min()}-{cnt.max()} "
          f"(uniform {len(R)//a.K}) | EFFECTIVE kinds {pr:.1f} of {a.K}", flush=True)
    if pr < 0.6 * a.K:
        print(f"[fit] WARNING: only {pr:.1f} kinds carry real mass. log K = "
              f"{np.log(a.K):.2f} is then an overstated ceiling and rare kinds are "
              f"unlearnable noise -- consider a smaller K.", flush=True)


def stage_assign(a):
    """Nearest frozen centroid for every labelled move, plus the per-position value
    spread that weights the loss."""
    ke = _engine(a.ref_ckpt, a.device)
    pool = _pool(a)
    C = np.load(os.path.join(a.out, "centroids.npz"))
    mu, sd, cen, K = C["mu"], C["sd"], C["C"], int(C["K"])
    W = C["W"] if "W" in C.files else None
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
                            dev=a.device, max_rows=a.assign_rows, pool=pool,
                            max_moves=a.max_moves, pv_samples=a.pv_samples)
        if r is None:
            continue
        R, ri, mj, p = r
        rmv_a = p["rmv"]                                    # decompress ONCE
        Zc = (R - mu) / sd
        if W is not None:
            Zc = Zc @ W
            Zc = Zc / (np.linalg.norm(Zc, axis=1, keepdims=True) + 1e-9)
        cn = cen / (np.linalg.norm(cen, axis=1, keepdims=True) + 1e-9)
        d2 = -(Zc @ cn.T)                       # negative cosine = cost
        kid = (sinkhorn_assign(d2) if a.balance
               else d2.argmin(1).astype(np.int8))
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
        d = np.load(s)          # the shard itself, for its own move ordering
        mv_a, nmv_a = d["mv"], d["n_mv"]                    # decompress ONCE
        MV48 = mv_a.shape[1]
        fk48 = np.full((n, MV48), -1, np.int8)
        miss = 0
        for i in range(n):
            k = int(nmv_a[i])
            if k == 0:
                continue
            exact, byft = {}, {}
            for c in range(k):
                key = (int(mv_a[i, c, 1]), int(mv_a[i, c, 2]),
                       int(mv_a[i, c, 5]))
                exact[key] = c
                byft.setdefault(key[:2], []).append(c)
            for j in range(NMV):
                if fkind[i, j] < 0:
                    continue
                f_, t_, pr = (int(x) for x in rmv_a[i, j])
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
        import time as _tw
        _w0 = _tw.time()
        np.savez_compressed(f + ".tmp.npz", **out)
        os.replace(f + ".tmp.npz", f)
        print(f"[wr] {os.path.basename(s)}: remap+write {_tw.time()-_w0:.0f}s", flush=True)
        done += 1
        # report over LABELLED rows only. Reporting over all 2048 conflates "no
        # choice here" with "not labelled yet" and reads as a collapsed clustering
        # when it is only --assign-rows doing its job.
        lab = (fk48 >= 0).any(-1)
        nk = np.array([len(np.unique(fk48[i][fk48[i] >= 0]))
                       for i in np.nonzero(lab)[0]]) if lab.any() else np.zeros(1)
        print(f"[assign] {os.path.basename(s)}: {int((fk48>=0).sum())} moves "
              f"mapped ({miss} unmapped) over {int(lab.sum())} rows | "
              f"{nk.mean():.1f} distinct kinds/row | "
              f"{100*np.mean(spread[lab] > 0.25):.0f}% of LABELLED rows have a choice "
              f"worth >0.25 pawns (mean {spread[lab][spread[lab]>0].mean():.2f})",
              flush=True)
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
    b1 = encode_boards(ke, boards, bs=1, dev=a.device)
    e1 = np.abs(b1 - ref).max()
    # The exactness claim belongs to CPU. MPS does not reproduce CPU kernels
    # bit-for-bit (measured: 0.030% of cells differ, per-state cosine 1.000000), so
    # asserting equality there would be a threshold carried into a context it was
    # never calibrated for. On mps the meaningful claim is directional: the labels are
    # an argmax of cosine against K centroids, so what must hold is that the state
    # DIRECTION is preserved.
    on_cpu = str(getattr(ke, "dev", "cpu")).startswith("cpu")
    if on_cpu:
        assert e1 == 0.0, f"batch-1 path is NOT the single-board path: {e1}"
    else:
        cos1 = float((b1 * ref).sum(1).min() /
                     (np.linalg.norm(b1, axis=1) * np.linalg.norm(ref, axis=1)
                      + 1e-9).min())
        assert cos1 > 0.9999, f"batch-1 direction diverges on {ke.dev}: cos {cos1}"
    dif = np.abs(encode_boards(ke, boards, bs=8, dev=a.device) - ref)
    frac = float((dif > 1e-4).mean())
    assert frac < 1e-3, f"batching changes {100*frac:.3f}% of cells -- too many"
    print(f"[gate] batch-1 {'EXACT (cpu)' if on_cpu else 'direction-preserving (mps)'}; "
          f"batch-8 differs in {100*frac:.3f}% of cells (max {dif.max():.1e}) -- PASS",
          flush=True)


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
    ap.add_argument("--K", type=int, default=8,
                    help="pick this from the fit's EFFECTIVE-kinds "
                         "report, not by taste: at K=12 on a 686-move "
                         "sample only 5.6 kinds carried mass. --refit "
                         "re-clusters from cache for free.")
    ap.add_argument("--n-pca", type=int, default=64)
    ap.add_argument("--balance", action="store_true", default=True,
                    help="Sinkhorn-Knopp equipartition (SwAV/SeLa). On "
                         "by default: nearest-centroid put 48%% of moves "
                         "in one kind.")
    ap.add_argument("--no-balance", dest="balance",
                    action="store_false")
    ap.add_argument("--refit", action="store_true",
                    help="re-cluster from the cached residuals; free")
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--fit-shards", type=int, default=6)
    ap.add_argument("--rows-per-shard", type=int, default=400)
    ap.add_argument("--assign-shards", type=int, default=50)
    ap.add_argument("--max-moves", type=int, default=0,
                    help="label only the top-N moves (0 = all 12). Beam 8 "
                         "already retains a winning move 94.7%% of the time.")
    ap.add_argument("--pv-samples", type=int, default=0,
                    help="encode only N plies per line, stratified across "
                         "the whole line (0 = every ply). Keeps the "
                         "horizon, cuts the cost.")
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
