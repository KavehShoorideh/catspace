#!/usr/bin/env python
"""experiments/bootstrap_mate_engine.py -- BOOTSTRAP MATE ENGINE (Kaveh 2026-07-24): NO
external mate bank. The engine starts knowing nothing; MCTS (energy prior + mate_stop) probes
the reachability field, every checkmate LEAF the search touches is harvested into an online
bank (own experience only), and the value becomes distance-to-DISCOVERED-mates. One knob:
search budget (--nodes). Question: at what budget does KRRvK-central hit 100%?

Fast path: priors cached by position (stable net), F-embeddings cached by position (stable
towers), only the bank-min-distance recomputed as the bank grows (cheap). Field net on MPS.
Games parallelized across workers sharing discoveries via an append-only FEN file.

Run (launcher spawns workers):   --nodes 5000 --n 48 --j 4
Single worker (internal):        --nodes 5000 --n 48 --j 4 --worker 0
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.engine.fields import FieldModel
from catspace.nn.mcts import MCTS
from catspace.tb import TB, tb_best_move
from experiments.mate_ladder_eval import sample_scenarios


class OnlineMateBank:
    """Episodic memory of mates the ENGINE found (rules-certified terminal states). Shared
    across workers via an append-only FEN file; embeddings computed locally, deduped by EPD."""

    def __init__(self, fm: FieldModel, bank_file: Path):
        self.fm = fm; self.bank_file = bank_file
        self.keys: set[str] = set(); self.embs: np.ndarray | None = None

    def _embed_add(self, boards):
        E = self.fm.embed_B_boards(boards)
        self.embs = E if self.embs is None else np.concatenate([self.embs, E])

    def sync(self):
        """pick up other workers' discoveries (eventually-consistent shared memory)."""
        if not self.bank_file.exists():
            return
        new = []
        for line in self.bank_file.read_text().splitlines():
            epd = line.strip()
            if epd and epd not in self.keys:
                self.keys.add(epd); new.append(chess.Board(epd))
        if new:
            self._embed_add(new)

    def add(self, boards) -> int:
        fresh = []
        for b in boards:
            epd = b.epd()
            if epd not in self.keys:
                self.keys.add(epd); fresh.append(b)
        if fresh:
            self._embed_add(fresh)
            with open(self.bank_file, "a") as f:
                f.writelines(b.epd() + "\n" for b in fresh)
        return len(fresh)

    def __len__(self):
        return len(self.keys)


class MilestoneCache:
    """Positions the engine has actually SEARCHED, with how well that went (own experience,
    Kaveh 2026-07-24 'milestone cache'): per position -- games through it, wins, searches
    that saw mate in-tree, nodes spent. Observation lines shared append-only across workers
    (aggregate rebuilt on sync; idempotent). B-embeddings local, computed once per epd.
    p_win = (wins+1)/(tries+2) (Beta-smoothed). Recording only changes MEMORY, not play;
    wiring into value/budget is a separate, flag-gated decision."""

    def __init__(self, fm: FieldModel, path: Path):
        self.fm = fm; self.path = path
        self.stats: dict[str, list] = {}          # epd -> [tries, wins, mate_seen, nodes]
        self.embs: np.ndarray | None = None; self._order: list[str] = []
        self.sync()

    def _ensure_emb(self, epds):
        new = [e for e in epds if e not in self._order]
        if new:
            E = self.fm.embed_B_boards([chess.Board(e) for e in new])
            self.embs = E if self.embs is None else np.concatenate([self.embs, E])
            self._order.extend(new)

    def sync(self):
        if not self.path.exists():
            return
        agg: dict[str, list] = {}
        for ln in self.path.read_text().splitlines():
            try:
                epd, mated, mseen, nodes = ln.rsplit(",", 3)
                s = agg.setdefault(epd, [0, 0, 0, 0])
                s[0] += 1; s[1] += int(mated); s[2] += int(mseen); s[3] += int(nodes)
            except ValueError:
                continue
        self.stats = agg
        self._ensure_emb(list(agg))

    def record_game(self, epds, mated, mate_seen_flags, nodes_per_move):
        with open(self.path, "a") as f:
            for e, ms, nd in zip(epds, mate_seen_flags, nodes_per_move):
                f.write(f"{e},{int(mated)},{int(ms)},{nd}\n")
                s = self.stats.setdefault(e, [0, 0, 0, 0])
                s[0] += 1; s[1] += int(mated); s[2] += int(ms); s[3] += nd
        self._ensure_emb(epds)

    def p_win(self, epd) -> float:
        s = self.stats.get(epd)
        return 0.5 if s is None else (s[1] + 1) / (s[0] + 2)

    def query(self, boards):
        """per board: (distance to nearest milestone, that milestone's p_win) -- the
        primitive both future wirings (value steering / budget allocation) need."""
        import torch
        if not self._order:
            return np.full(len(boards), np.inf), np.full(len(boards), 0.5)
        F = self.fm.embed_F_boards(boards)
        bt = torch.from_numpy(self.embs).to(self.fm.device)
        best_d = np.full(len(F), np.inf); best_i = np.zeros(len(F), np.int64)
        for s in range(0, len(F), self.fm.chunk):
            with torch.no_grad():
                D = self.fm.fb.distance_matrix(
                    torch.from_numpy(F[s:s + self.fm.chunk]).to(self.fm.device), bt)
                m, a = D.min(1)
            best_d[s:s + len(m)] = m.cpu().numpy(); best_i[s:s + len(m)] = a.cpu().numpy()
        p = np.array([self.p_win(self._order[i]) for i in best_i])
        return best_d, p


def harvest(root) -> tuple[list, list]:
    """(win_mates, loss_mates) among checkmate leaves the search TOUCHED:
    Black to move & mated = White wins; White to move & mated = White LOSES
    (possible once promotions exist, e.g. KRRvKP lines -- feeds the loss bank)."""
    wins, losses, stack = [], [], [root]
    while stack:
        n = stack.pop()
        if n.board is not None and n.board.is_checkmate():
            (wins if n.board.turn == chess.BLACK else losses).append(n.board)
        stack.extend(n.children)
    return wins, losses


def make_boot_value(fm: FieldModel, bank: OnlineMateBank, times: dict | None = None,
                    loss_bank: OnlineMateBank | None = None,
                    dtm_ckpt: str | None = None, nucleus_max_pieces: int = 5):
    """WDL leaf value (Kaveh 2026-07-25 'I want it'): three-outcome Boltzmann from the
    field's energies to the two DISCOVERED absorbing regions plus a draw mass --
        p_w ~ exp(-d_win/M)   p_l ~ exp(-d_loss/M)   p_d ~ kappa = e^-1
        v = (p_w - p_l) / (p_w + p_l + p_d)   in (-1, 1)
    M = running median of d_win (temperature, NOT a center -- the old tanh((M-d)/M) was
    secretly this formula with a PHANTOM loss target planted at distance M, which is why
    draws outranked median-distance winning positions). One-hot terminals under w-l:
    mate=+1, draw=0, mated=-1 -- and w+d/2 is affine in w-l, so the default
    expected-points objective needs no MCTS change. Non-default functionals (must-win w,
    draw-ok w+d) need node-level triples: documented follow-on.
    Empty loss bank (no discovered threats) reduces to v = p_w/(p_w+kappa) in (0,1):
    every live position outranks a draw. Loss threats push v below 0 -> the rules-exact
    draw terminals dominate -> the engine SEEKS stalemate/repetition when lost.

    dmin caching (Kaveh 2026-07-24): banks only GROW, so min-distance decomposes
    exactly -- dmin_new = min(dmin_cached, d(x, bank[ver:])). Each position pays for the
    bank TAIL added since its last query, not the full bank."""
    emb_cache: dict[str, np.ndarray] = {}
    recent = deque(maxlen=512)
    KAPPA = float(np.exp(-1.0))

    # LAST MILE (Kaveh 2026-07-25): humans RESIGN trivially-won endings, so the field's
    # behavior-geometry has no trajectory support in the nucleus -- its distances there
    # are extrapolations (measured: value INVERSIONS at cycle positions). Inside the
    # tb nucleus (<= nucleus_max_pieces) the distance source becomes the DTM regression
    # trained OFFLINE on tablebase-optimal plies (outcome-structure facts; the engine
    # still never probes tb at play). Same WDL Boltzmann, real ply units, M=scale.
    dtm_net = None
    if dtm_ckpt:
        import torch
        from experiments.train_dtm_cnn import DTMNet
        from catspace.data.encode import encode_meta, encode_packed
        from catspace.nn.features import feature_planes
        st = torch.load(dtm_ckpt, map_location="cpu", weights_only=False)
        dtm_net = DTMNet(c=st["c"]).to(fm.device)
        dtm_net.load_state_dict(st["state"]); dtm_net.eval()
        dtm_scale = float(st.get("scale", 20.0))
        dtm_cache: dict[str, float] = {}

        def _dtm_hat(boards, keys):
            import torch as _t
            miss = [i for i, k in enumerate(keys) if k not in dtm_cache]
            if miss:
                tt = time.perf_counter()
                pk = np.stack([encode_packed(boards[i]) for i in miss])
                mt = np.stack([encode_meta(boards[i]) for i in miss])
                with _t.no_grad():
                    pred = dtm_net(_t.from_numpy(feature_planes(pk, mt)).to(fm.device)) \
                        .cpu().numpy() * dtm_scale
                for i, p in zip(miss, pred):
                    dtm_cache[keys[i]] = float(p)
                if times is not None:
                    times["dtm_s"] = times.get("dtm_s", 0.0) + time.perf_counter() - tt
                    times["dtm_n"] = times.get("dtm_n", 0) + len(miss)
            return np.array([dtm_cache[k] for k in keys])

    def _embed(boards, keys):
        miss = [b for b, k in zip(boards, keys) if k not in emb_cache]
        if miss:
            tt = time.perf_counter()
            E = fm.embed_F_boards(miss)
            for b, e in zip(miss, E):
                emb_cache[b.epd()] = e
            if times is not None:
                times["embedF_s"] = times.get("embedF_s", 0.0) + time.perf_counter() - tt
                times["embedF_n"] = times.get("embedF_n", 0) + len(miss)

    def _dmin_tracker(bk: OnlineMateBank):
        dmin_cache: dict[str, tuple[int, float]] = {}   # epd -> (bank_version, dmin)

        def dmin(keys):
            nb = len(bk)
            stale = [i for i, k in enumerate(keys) if dmin_cache.get(k, (0, np.inf))[0] < nb]
            if stale:
                tt = time.perf_counter()
                # group by EXACT cached version: one ancient position must not drag the
                # whole batch into a near-full-bank rescan (the 1430s-dbank pathology)
                by_ver: dict[int, list] = {}
                for i in stale:
                    by_ver.setdefault(dmin_cache.get(keys[i], (0, np.inf))[0], []).append(i)
                for ver, idxs in by_ver.items():
                    F = np.stack([emb_cache[keys[i]] for i in idxs])
                    d_tail = fm.d_to_bank(F, bk.embs[ver:])
                    for i, dt in zip(idxs, d_tail):
                        dmin_cache[keys[i]] = (nb, min(dmin_cache.get(keys[i], (0, np.inf))[1], float(dt)))
                if times is not None:
                    times["dbank_s"] = times.get("dbank_s", 0.0) + time.perf_counter() - tt
                    times["dbank_n"] = times.get("dbank_n", 0) + len(stale)
            return np.array([dmin_cache[k][1] for k in keys])
        return dmin

    dmin_win = _dmin_tracker(bank)
    dmin_loss = _dmin_tracker(loss_bank) if loss_bank is not None else None
    cand = {"idx": None, "ver": -1, "audit": 0}   # per-move candidate annulus (set_anchor)

    def set_anchor(board, slack=16.0):
        """Kaveh 2026-07-25 ('if moves are reversible, can we construct distance without
        recalculating?'): step-consistency (d_step~1/ply) + triangle inequality => a leaf
        d plies under the root can only reach bank points within dmin_root + ~d + margin.
        One full-bank query at the root per move; all leaves evaluate against the annulus
        d(root,g) <= dmin_root + slack only. Periodic exact-vs-pruned audit keeps it honest."""
        cand["cache"] = {}                       # approx dmins live within ONE move only
        if len(bank) < 256:
            cand["idx"] = None; return
        cand["moves"] = cand.get("moves", 0) + 1
        F = fm.embed_F_boards([board])
        # Kaveh 2026-07-25: triangle-inequality estimates BETWEEN refreshes, full
        # recalculation every 5 moves, error(est vs exact) reported at each refresh
        if cand["idx"] is not None and cand["moves"] % 5 != 1:
            return                               # carry candidates; leaves use idx + tail
        est = None
        if cand["idx"] is not None:
            sub = bank.embs[cand["idx"]]
            if len(bank) > cand["ver"]:
                sub = np.concatenate([sub, bank.embs[cand["ver"]:]])
            est = float(fm.d_to_bank(F, sub)[0])
        d_all = _bank_row(F, bank.embs)
        exact = float(d_all.min())
        if est is not None:
            print(f"    [tri-refresh] dmin est {est:.3f} exact {exact:.3f} "
                  f"err {est - exact:+.3f}", flush=True)
        thr = exact + slack
        idx = np.flatnonzero(d_all <= thr)
        if len(idx) < 128 or len(idx) > 256:
            # dense-bank regime: the annulus doesn't bite (mates cluster within ~slack of
            # any won position) -- cap at nearest-256 and let the printed audit police it
            idx = np.argsort(d_all)[:256]
        cand["idx"] = idx; cand["ver"] = len(bank); cand["audit"] += 1

    def _bank_row(F, embs):
        import torch
        out = None
        for bs in range(0, len(embs), 512):
            bt = torch.from_numpy(embs[bs:bs + 512]).to(fm.device)
            with torch.no_grad():
                m = fm.fb.distance_matrix(
                    torch.from_numpy(F).to(fm.device), bt)
            m = m.cpu().numpy()
            out = m if out is None else np.concatenate([out, m], axis=1)
        return out[0]

    def value_fn(boards):
        nw = len(bank); nl = len(loss_bank) if loss_bank is not None else 0
        if nw == 0 and nl == 0:
            return np.zeros(len(boards))
        keys = [b.epd() for b in boards]
        nuc = ([i for i, b in enumerate(boards) if len(b.piece_map()) <= nucleus_max_pieces]
               if dtm_net is not None else [])
        if len(nuc) == len(boards):
            # whole batch in the nucleus: DTM path only, no field/bank work at all
            d_w = _dtm_hat(boards, keys)
            p_w = np.exp(-d_w / dtm_scale)
            p_l = 0.0
            if nl:
                _embed(boards, keys)
                p_l = np.exp(-dmin_loss(keys) / dtm_scale)
            return (p_w - p_l) / (p_w + p_l + KAPPA)
        _embed(boards, keys)
        if nw and cand["idx"] is not None:
            # pruned path: candidates from the root annulus + any mates banked since the
            # anchor; move-local cache only (approx values never persist across moves)
            cc = cand.setdefault("cache", {})
            miss = [i for i, k in enumerate(keys) if k not in cc]
            if miss:
                tt = time.perf_counter()
                F = np.stack([emb_cache[keys[i]] for i in miss])
                sub = bank.embs[cand["idx"]]
                if len(bank) > cand["ver"]:
                    sub = np.concatenate([sub, bank.embs[cand["ver"]:]])
                d_sub = fm.d_to_bank(F, sub)
                if cand["audit"] % 25 == 1 and not cc:      # honesty audit, rare
                    d_ex = fm.d_to_bank(F, bank.embs)
                    print(f"    [prune-audit] max|err| {np.abs(d_sub - d_ex).max():.3f} "
                          f"cands {len(sub)}/{len(bank)}", flush=True)
                for i, dv in zip(miss, d_sub):
                    cc[keys[i]] = float(dv)
                if times is not None:
                    times["dbank_s"] = times.get("dbank_s", 0.0) + time.perf_counter() - tt
                    times["dbank_n"] = times.get("dbank_n", 0) + len(miss)
            d_w = np.array([cc[k] for k in keys])
        else:
            d_w = dmin_win(keys) if nw else None
        d_l = dmin_loss(keys) if nl else None
        recent.extend((d_w if d_w is not None else d_l).tolist())
        M = max(float(np.median(recent)), 1e-6)
        p_w = np.exp(-d_w / M) if d_w is not None else 0.0
        p_l = np.exp(-d_l / M) if d_l is not None else 0.0
        v = (p_w - p_l) / (p_w + p_l + KAPPA)
        if nuc:                                     # mixed batch: nucleus rows -> DTM value
            db = [boards[i] for i in nuc]; dk = [keys[i] for i in nuc]
            pw_n = np.exp(-_dtm_hat(db, dk) / dtm_scale)
            pl_n = (np.exp(-np.asarray(d_l)[nuc] / dtm_scale) if d_l is not None
                    else np.zeros(len(nuc)))
            v[nuc] = (pw_n - pl_n) / (pw_n + pl_n + KAPPA)
        return v
    value_fn.set_anchor = set_anchor
    value_fn.invalidate_anchor = lambda: cand.update(idx=None)   # irreversible move played:
    return value_fn                                              # the +-1-ply bound is void


def make_batched_energy_prior(ckpt: str, cohort: int = 11, device: str = "cpu",
                              times: dict | None = None):
    """(policy_fn, policy_batch_fn) sharing one net + one epd cache. The batch fn is the
    speed path: one forward for a whole leaf batch instead of ~2ms singles (mcts.py already
    supports policy_batch_fn -- no search-semantics change, same net, same numbers)."""
    import torch
    from catspace.data.encode import encode_meta, encode_packed
    from catspace.nn.features import feature_planes
    from catspace.nn.fb import pick_device
    from catspace.nn.opponent import OpponentModel
    dev = pick_device(device)
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = OpponentModel(**st["config"]).to(dev)
    net.load_state_dict(st["state"]); net.eval()
    L = st["config"].get("max_moves", 80)
    cache: dict[str, dict] = {}

    def _forward(boards):
        tt = time.perf_counter()
        B = len(boards)
        f = np.zeros((B, L), np.int64); t = np.zeros((B, L), np.int64)
        pc = np.zeros((B, L), np.int64); ct = np.zeros((B, L), np.int64)
        nm = np.zeros(B, np.int64); movess = []
        for i, b in enumerate(boards):
            moves = list(b.legal_moves)[:L]; movess.append(moves); nm[i] = len(moves)
            for j, m in enumerate(moves):
                f[i, j], t[i, j] = m.from_square, m.to_square
                pc[i, j] = b.piece_type_at(m.from_square) or 0
                cap = b.piece_type_at(m.to_square)
                ct[i, j] = cap or (1 if b.is_en_passant(m) else 0)
        pk = np.stack([encode_packed(b) for b in boards])
        mt = np.stack([encode_meta(b) for b in boards])
        pl = torch.from_numpy(feature_planes(pk, mt)).to(dev)
        with torch.no_grad():
            lg = net(pl, torch.from_numpy(f).to(dev), torch.from_numpy(t).to(dev),
                     torch.from_numpy(pc).to(dev), torch.from_numpy(ct).to(dev),
                     torch.from_numpy(nm).to(dev),
                     torch.full((B,), cohort, dtype=torch.int64).to(dev))
        out = []
        for i, moves in enumerate(movess):
            p = torch.softmax(lg[i, :len(moves)], 0).cpu().numpy()
            out.append({m: float(p[j]) for j, m in enumerate(moves)})
        if times is not None:
            times["prior_s"] = times.get("prior_s", 0.0) + time.perf_counter() - tt
            times["prior_n"] = times.get("prior_n", 0) + B
        return out

    def policy_fn(b):
        k = b.epd()
        if k not in cache:
            cache[k] = _forward([b])[0]
        return cache[k]

    def policy_batch_fn(boards):
        keys = [b.epd() for b in boards]
        miss_i = [i for i, k in enumerate(keys) if k not in cache]
        if miss_i:
            for i, pri in zip(miss_i, _forward([boards[i] for i in miss_i])):
                cache[keys[i]] = pri
        return [cache[k] for k in keys]

    return policy_fn, policy_batch_fn


def tb_white_move(b, tb):
    """win-preserving min-|dtz| move that leaves Black NO immediate draw claim (announce
    rule); falls back to best dtz if every winning move is claimable. Consulted only when
    the field has no gradient; every use is LOGGED (wins must stay attributable)."""
    alts = []
    for m in b.legal_moves:
        c = b.copy(stack=False); c.push(m)
        w, dz = tb.wdl_dtz(c)
        if w is not None and -w == 2:
            alts.append((abs(dz) if dz is not None else 999, m))
    for _, m in sorted(alts, key=lambda x: x[0]):
        b.push(m)
        ok = not b.can_claim_threefold_repetition()
        b.pop()
        if ok:
            return m
    return sorted(alts, key=lambda x: x[0])[0][1] if alts else None


def worker(args):
    t0 = time.time(); tb = TB()
    starts = dict(sample_scenarios(np.random.default_rng(args.seed), args.n))[args.scenario]
    fm = FieldModel(args.field, device=args.device)
    bank = OnlineMateBank(fm, Path(args.bank_file))
    loss_bank = OnlineMateBank(fm, Path(args.loss_bank_file))
    times: dict = {}
    vfn = make_boot_value(fm, bank, times, loss_bank,
                          dtm_ckpt=args.last_mile_dtm or None,
                          nucleus_max_pieces=args.nucleus_max_pieces)
    pfn, pfnb = make_batched_energy_prior(args.energy_ckpt, device="cpu", times=times)
    ms = MilestoneCache(fm, Path(args.milestone_file))

    res_path = Path(args.results_file)
    done = set()
    if res_path.exists():
        import json
        done = {json.loads(ln)["g"] for ln in res_path.read_text().splitlines() if ln.strip()}
    base = ([int(x) for x in args.games.split(",")] if args.games
            else list(range(len(starts))))
    my_games = [g for g in base[args.worker::args.j] if g not in done]
    if done:
        print(f"[worker {args.worker}] resume: skipping {len(done)} recorded games", flush=True)
    results = []
    for gi in my_games:
        bank.sync(); loss_bank.sync(); ms.sync()
        b = starts[gi].copy(stack=False)
        start_epd = b.epd()
        plies = 0; nodes_spent = 0; tmoves = []; found_this_game = 0
        roots: list[str] = []; mseen: list[bool] = []; nmoves: list[int] = []
        ucis: list[str] = []    # full trajectory (start_epd + ucis reproduces the game)
        tb_consults: list[int] = []   # plies where the tb fallback fired (attribution log)
        from collections import Counter as _Counter
        hist = _Counter({b.epd(): 1})   # ALL position visits, both colors (stuckness +
                                        # repetition-creation triggers); counted at arrival
        tb_mode = False   # STICKY handover (g036: alternating field/tb control oscillates
                          # through the consulted position into threefold; once the field
                          # proves gradient-less in a game, tb converts the rest, all logged)
        reuse = None            # subtree carried across moves (tree reuse; general lever)
        # WARM-UP (Kaveh 2026-07-25 'run until bank is full on first move'): before the
        # first timed move, keep searching+harvesting until the bank reaches the target
        # (cap: 20x one move's budget). Warm trees feed the real move via reuse.
        if args.warm_bank > 0 and len(bank) < args.warm_bank and b.turn == chess.WHITE:
            tw = time.time(); warm_nodes = 0; warm0 = len(bank)
            while len(bank) < args.warm_bank and warm_nodes < 20 * args.nodes:
                if hasattr(vfn, "set_anchor"):
                    vfn.set_anchor(b)
                mw = MCTS(lambda bs: np.zeros(len(bs)), max_nodes=args.nodes, mate_stop=True,
                          pw_c=1.5, root_min_visits=10, value_fn=vfn, policy_fn=pfn,
                          policy_batch_fn=pfnb, batch_leaves=32)
                wroot = mw.run(b, reuse_root=reuse)
                w_w, w_l = harvest(wroot)
                bank.add(w_w)
                if w_l:
                    loss_bank.add(w_l)
                warm_nodes += mw.evals_used
                reuse = wroot
                if mw.evals_used <= 1:          # mate at root: no more warming needed
                    break
            print(f"    [warm-up] bank {warm0}->{len(bank)} nodes={warm_nodes} "
                  f"[{time.time()-tw:.0f}s]", flush=True)
        # asymmetric claiming (g041 harness artifact: outcome(claim_draw=True) auto-claims
        # for EITHER side -- White would never claim a draw in a won position; only the
        # DEFENDER claims). Automatic draws (75-move/fivefold/stalemate/material) still end.
        def _game_over():
            if b.is_game_over():
                return True
            return b.turn == chess.BLACK and (b.can_claim_threefold_repetition()
                                              or b.can_claim_fifty_moves())
        while plies < args.max_plies and not _game_over():
            if b.turn == chess.WHITE:
                # STUCKNESS trigger (Kaveh 'do the fix'): second visit to a position =
                # the field has no EFFECTIVE gradient in play (confidently-wrong loops
                # never trip the flatness trigger) -> consult tb directly, LOGGED.
                if (args.tb_fallback_eps > 0
                        and (tb_mode or hist[b.epd()] >= 2 or b.halfmove_clock >= 60
                             or plies >= 60)
                        and len(b.piece_map()) <= 6):
                    mv_tb = tb_white_move(b, tb)
                    if mv_tb is not None:
                        tb_mode = True
                        tb_consults.append(plies)
                        roots.append(b.epd()); mseen.append(False); nmoves.append(0)
                        tmoves.append(0.0); ucis.append(mv_tb.uci())
                        if (b.is_capture(mv_tb) or b.piece_type_at(mv_tb.from_square) == chess.PAWN) \
                                and hasattr(vfn, "invalidate_anchor"):
                            vfn.invalidate_anchor()
                        b.push(mv_tb)
                        hist[b.epd()] += 1
                        reuse = None
                        plies += 1
                        continue
                tm = time.time(); snap = dict(times)
                m = MCTS(lambda bs: np.zeros(len(bs)), max_nodes=args.nodes, mate_stop=True,
                         pw_c=1.5, root_min_visits=10, value_fn=vfn, policy_fn=pfn,
                         policy_batch_fn=pfnb, batch_leaves=32)
                roots.append(b.epd())
                if hasattr(vfn, "set_anchor"):
                    vfn.set_anchor(b)
                if reuse is not None:
                    reuse.parent = None     # detach: stale ancestors skew the mate-depth
                                            # discount and double-count _threefold's walk
                root = m.run(b, reuse_root=reuse)
                t_search = time.time() - tm
                th = time.perf_counter()
                win_mates, loss_mates = harvest(root)
                mseen.append(len(win_mates) > 0); nmoves.append(m.evals_used)
                found_this_game += bank.add(win_mates)
                if loss_mates:
                    loss_bank.add(loss_mates)
                t_harv = time.perf_counter() - th
                best = max(root.children,
                           key=lambda c: (c.N, (c.terminal_v if c.terminal_v is not None else c.Q)))
                nodes_spent += m.evals_used; tmoves.append(time.time() - tm)
                # TB FALLBACK (Kaveh): if the searched root shows NO gradient (children
                # value spread < eps) and the position is tb-probeable, consult tb and LOG.
                if args.tb_fallback_eps > 0 and len(b.piece_map()) <= 6 and root.children:
                    qs = [c.terminal_v if c.terminal_v is not None
                          else (c.W / c.N if c.N > 0 else None) for c in root.children]
                    qs = [q for q in qs if q is not None]
                    if len(qs) >= 2 and (max(qs) - min(qs)) < args.tb_fallback_eps:
                        mv_tb = tb_white_move(b, tb)
                        if mv_tb is not None:
                            best = next((c for c in root.children if c.move == mv_tb), best)
                            tb_consults.append(plies)
                # repetition-CREATION veto: if the chosen move pushes into a position
                # already seen twice (threefold = instant draw claim), consult tb instead
                if args.tb_fallback_eps > 0 and len(b.piece_map()) <= 6:
                    nb2 = b.copy(stack=False); nb2.push(best.move)
                    if hist[nb2.epd()] >= 2:
                        mv_tb = tb_white_move(b, tb)
                        if mv_tb is not None and mv_tb != best.move:
                            best = next((c for c in root.children if c.move == mv_tb), best)
                            tb_mode = True
                            tb_consults.append(plies)
                d = {k: times.get(k, 0) - snap.get(k, 0) for k in
                     ("prior_s", "prior_n", "embedF_s", "embedF_n", "dbank_s", "dbank_n")}
                tree = t_search - d["prior_s"] - d["embedF_s"] - d["dbank_s"]
                print(f"    mv{len(tmoves):02d} {tmoves[-1]:6.1f}s = prior {d['prior_s']:5.1f} "
                      f"({d['prior_n']:4d}) + embF {d['embedF_s']:5.1f} ({d['embedF_n']:4d}) "
                      f"+ dbank {d['dbank_s']:5.1f} ({d['dbank_n']:5d}) + tree {tree:5.1f} "
                      f"+ harvest {t_harv:4.1f}  nodes={m.evals_used}", flush=True)
                # ANNOUNCE-rule guard (g017: python-chess honors FIDE's claim-by-announcing
                # -- Black can claim a threefold it never plays out; arrival counters are
                # blind to it). If the chosen move leaves Black an immediate claim, walk
                # the tb-winning moves by dtz until one is claim-safe.
                mv_final = best.move
                if args.tb_fallback_eps > 0 and len(b.piece_map()) <= 6:
                    b.push(mv_final)
                    unsafe = b.can_claim_threefold_repetition()
                    b.pop()
                    if unsafe:
                        alts = []
                        for m2 in b.legal_moves:
                            c2 = b.copy(stack=False); c2.push(m2)
                            w2, dz2 = tb.wdl_dtz(c2)
                            if w2 is not None and -w2 == 2:
                                alts.append((abs(dz2) if dz2 is not None else 999, m2))
                        for _, m2 in sorted(alts, key=lambda x: x[0]):
                            b.push(m2)
                            ok = not b.can_claim_threefold_repetition()
                            b.pop()
                            if ok:
                                mv_final = m2
                                tb_mode = True
                                tb_consults.append(plies)
                                best = next((c for c in root.children if c.move == m2), best)
                                break
                ucis.append(mv_final.uci())
                if (b.is_capture(mv_final) or b.piece_type_at(mv_final.from_square) == chess.PAWN) \
                        and hasattr(vfn, "invalidate_anchor"):
                    vfn.invalidate_anchor()      # irreversible: candidate set is void
                b.push(mv_final)
                hist[b.epd()] += 1               # count BOTH colors (route-independent
                if hist[b.epd()] >= 2:           # repetitions live on Black-side keys too)
                    tb_mode = True               # g043: BLACK completes threefolds -- any
                reuse = best if best.move == mv_final else None   # 2nd occurrence => tb
            else:
                mvb = tb_best_move(b, tb)
                if reuse is not None:
                    reuse = next((c for c in reuse.children if c.move == mvb), None)
                if (b.is_capture(mvb) or b.piece_type_at(mvb.from_square) == chess.PAWN) \
                        and hasattr(vfn, "invalidate_anchor"):
                    vfn.invalidate_anchor()
                ucis.append(mvb.uci())
                b.push(mvb)
                hist[b.epd()] += 1
                if hist[b.epd()] >= 2:
                    tb_mode = True
            plies += 1
        out = b.outcome(claim_draw=True)
        mated = bool(out and out.winner == chess.WHITE)
        term = out.termination.name.lower() if out else "timeout"
        ms.record_game(roots, mated, mseen, nmoves)
        import json
        rec = dict(g=gi, mate=mated, term=term, plies=plies, nodes=nodes_spent,
                   t=round(sum(tmoves), 1), moves=len(tmoves), bank=len(bank),
                   tb_consults=tb_consults)
        if not mated:           # FAILs carry the full trajectory for field diagnostics
            rec["start_epd"] = start_epd; rec["ucis"] = ucis
        with open(res_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        results.append((gi, mated, plies, nodes_spent, sum(tmoves), len(tmoves)))
        print(f"  g{gi:03d}[w{args.worker}] {'mate' if mated else 'FAIL:' + term} plies={plies} "
              f"tb={len(tb_consults)} bank={len(bank)}(+{found_this_game}) "
              f"loss={len(loss_bank)} ms={len(ms.stats)} "
              f"t/move={np.median(tmoves):.1f}s t/game={sum(tmoves):.0f}s "
              f"nodes/s={nodes_spent/max(sum(tmoves),1e-9):.0f} [{time.time()-t0:.0f}s]", flush=True)
    tb.close()
    m_ = [r for r in results if r[1]]
    print(f"[worker {args.worker}] {len(m_)}/{len(results)} mate  "
          f"med t/move={np.median([t/max(k,1) for _, _, _, _, t, k in results]):.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", type=int, default=5000)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--j", type=int, default=4)
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--scenario", default="KRRvK-central")
    ap.add_argument("--field", default="data/derived/sep/lichess_mc2.pt")
    ap.add_argument("--energy-ckpt", default="data/derived/sep/opponent_energy_v1.pt")
    ap.add_argument("--bank-file", default=None)
    ap.add_argument("--loss-bank-file", default=None)
    ap.add_argument("--milestone-file", default=None)
    ap.add_argument("--results-file", default=None)
    ap.add_argument("--fresh", action="store_true",
                    help="wipe bank/milestones/results; DEFAULT resumes (checkpointed runs)")
    ap.add_argument("--games", default=None,
                    help="comma list of game indices to (re)play (default: all 0..n-1)")
    ap.add_argument("--warm-bank", type=int, default=1000,
                    help="first-move warm-up: search+harvest until the bank has this many "
                         "mates (cap 20x --nodes); 0 = off")
    ap.add_argument("--last-mile-dtm", default="data/derived/sep/dtm_cnn_v2.pt",
                    help="tb-trained DTM regression as the value's distance source INSIDE "
                         "the nucleus (resignation gap: the field has no trajectory support "
                         "there); '' = off")
    ap.add_argument("--nucleus-max-pieces", type=int, default=5)
    ap.add_argument("--tb-fallback-eps", type=float, default=0.02,
                    help="if the searched root's child-value spread < eps (no field "
                         "gradient) and <=6 pieces: consult tb, LOG the consult (win "
                         "attribution); 0 = never consult")
    ap.add_argument("--max-plies", type=int, default=80)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tag = f"n{args.nodes}_{args.scenario}"
    if args.bank_file is None:
        args.bank_file = f"artifacts/experiments/boot_bank_{tag}.fens"
    if args.loss_bank_file is None:
        args.loss_bank_file = f"artifacts/experiments/boot_lossbank_{tag}.fens"
    if args.milestone_file is None:
        args.milestone_file = f"artifacts/experiments/boot_milestones_{tag}.jsonl"
    if args.results_file is None:
        args.results_file = f"artifacts/experiments/boot_results_{tag}.jsonl"

    if args.worker is not None:
        worker(args); return

    if args.fresh:
        for p in (args.bank_file, args.loss_bank_file, args.milestone_file, args.results_file):
            Path(p).unlink(missing_ok=True)
    t0 = time.time()
    procs = [subprocess.Popen([sys.executable, __file__, *sys.argv[1:], "--worker", str(w)])
             for w in range(args.j)]
    for p in procs:
        p.wait()
    import json
    raw = [json.loads(ln) for ln in Path(args.results_file).read_text().splitlines() if ln.strip()] \
        if Path(args.results_file).exists() else []
    first = {}
    for r in raw:                       # dedup by game id, first occurrence wins (crash-replay safety)
        first.setdefault(r["g"], r)
    rows = list(first.values())
    n_bank = len(set(Path(args.bank_file).read_text().splitlines())) if Path(args.bank_file).exists() else 0
    m = [r for r in rows if r["mate"]]
    tba = [r for r in m if r.get("tb_consults")]
    tpm = [r["t"] / max(r["moves"], 1) for r in rows]
    print(f"VERDICT BOOTSTRAP_MATE scenario={args.scenario} nodes={args.nodes} "
          f"mate={len(m)}/{len(rows)} ({len(m)/max(len(rows),1):.2f}) "
          f"[clean={len(m)-len(tba)} tb-assisted={len(tba)}]  "
          f"med_plies={np.median([r['plies'] for r in m]) if m else float('nan'):.0f}  "
          f"bank_final={n_bank}  med_t/move={np.median(tpm) if tpm else float('nan'):.1f}s  "
          f"med_t/solve={np.median([r['t'] for r in m]) if m else float('nan'):.0f}s  "
          f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
