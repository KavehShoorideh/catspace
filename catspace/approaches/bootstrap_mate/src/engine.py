"""catspace/approaches/bootstrap_mate/src/engine.py -- the BOOTSTRAP MATE ENGINE glue
(Kaveh 2026-07-24; relocated from experiments/ in the 2026-08-03 restructure).

NO external mate bank. The engine starts knowing nothing; MCTS (energy prior + mate_stop)
probes the reachability field, every checkmate LEAF the search touches is harvested into an
online bank (own experience only), and the value becomes distance-to-DISCOVERED-mates.

This module is SHIPPING code: both deployment shells (the UCI engine and the HTTP
co-analyst) build their engine from it. The experiment driver that sweeps search budgets
lives beside it in ../experiments/bootstrap_mate_engine.py.
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import chess
import numpy as np

from catspace.fields import FieldModel
from catspace.research.components.planner.approaches.endgame_groundtruth.src.material import mat_sig

__all__ = ["OnlineMateBank", "MilestoneCache", "harvest", "make_boot_value", "make_planner",
           "make_batched_energy_prior", "tb_white_move", "mat_sig"]


from catspace.research.components.planner.approaches.endgame_groundtruth.src.material import mat_sig  # component home (refactor 2026-07-30)


class OnlineMateBank:
    """Episodic memory of mates the ENGINE found (rules-certified terminal states). Shared
    across workers via an append-only FEN file; embeddings computed locally, deduped by EPD."""

    def __init__(self, fm: FieldModel, bank_file: Path):
        self.fm = fm; self.bank_file = bank_file
        self.keys: set[str] = set(); self.embs: np.ndarray | None = None
        self.sigs: list[str] = []      # material class per exemplar (planner's density index)

    def _embed_add(self, boards):
        E = self.fm.embed_B_boards(boards)
        self.embs = E if self.embs is None else np.concatenate([self.embs, E])
        self.sigs.extend(mat_sig(b) for b in boards)

    def class_idx(self, sig: str) -> np.ndarray:
        return np.flatnonzero(np.array(self.sigs) == sig) if self.sigs else np.array([], int)

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


def harvest(root) -> tuple[list, list, list]:
    """(win_mates, loss_mates, stalemates) among terminal leaves the search TOUCHED.
    Mates: Black to move & mated = White wins; White mated = loss bank. Stalemates are
    the FIXED (history-free) component of the draw surface (Kaveh: 'draw bank') --
    bankable exactly like mates; the moving components (repetition, clock) are rules-
    computable per ply and enter kappa(state) directly, no bank needed."""
    wins, losses, stales, stack = [], [], [], [root]
    while stack:
        n = stack.pop()
        if n.board is not None:
            if n.board.is_checkmate():
                (wins if n.board.turn == chess.BLACK else losses).append(n.board)
            elif n.board.is_stalemate():
                stales.append(n.board)
        stack.extend(n.children)
    return wins, losses, stales


def make_boot_value(fm: FieldModel, bank: OnlineMateBank, times: dict | None = None,
                    loss_bank: OnlineMateBank | None = None,
                    dtm_ckpt: str | None = None, nucleus_max_pieces: int = 5,
                    draw_bank: OnlineMateBank | None = None,
                    game_ctx: dict | None = None):
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
    recent_l = deque(maxlen=512)     # loss-channel temperature is SELF-calibrated too:
    KAPPA = float(np.exp(-1.0))      # field-scale d_loss (~50) over dtm_scale (20) made
                                     # every threat read as ~nothing (the units bug)

    # LAST MILE (Kaveh 2026-07-25): humans RESIGN trivially-won endings, so the field's
    # behavior-geometry has no trajectory support in the nucleus -- its distances there
    # are extrapolations (measured: value INVERSIONS at cycle positions). Inside the
    # tb nucleus (<= nucleus_max_pieces) the distance source becomes the DTM regression
    # trained OFFLINE on tablebase-optimal plies (outcome-structure facts; the engine
    # still never probes tb at play). Same WDL Boltzmann, real ply units, M=scale.
    dtm_net = None
    if dtm_ckpt:
        import torch
        from catspace.research.tools.chess_specific.chessdata.encode import encode_meta, encode_packed
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes
        st = torch.load(dtm_ckpt, map_location="cpu", weights_only=False)
        dtm_scale = float(st.get("scale", 20.0))
        dtm_cache: dict[str, float] = {}
        is_tok = isinstance(st.get("config"), dict) and "heads" in st["config"]
        if is_tok:                                      # token transformer (Kaveh: no CNN)
            from experiments.train_dtm_tok import DTMTok, tokenize
            dtm_net = DTMTok(**st["config"]).to(fm.device)
        else:                                           # legacy CNN ckpt (until v4 lands)
            from experiments.train_dtm_cnn import DTMNet
            dtm_net = DTMNet(c=st["c"]).to(fm.device)
        dtm_net.load_state_dict(st["state"]); dtm_net.eval()

        def _dtm_hat(boards, keys):
            import torch as _t
            miss = [i for i, k in enumerate(keys) if k not in dtm_cache]
            if miss:
                tt = time.perf_counter()
                pk = np.stack([encode_packed(boards[i]) for i in miss])
                mt = np.stack([encode_meta(boards[i]) for i in miss])
                with _t.no_grad():
                    if is_tok:
                        pid, sqi, pad = tokenize(pk, mt)
                        pred = dtm_net(_t.from_numpy(pid).to(fm.device),
                                       _t.from_numpy(sqi).to(fm.device),
                                       _t.from_numpy(pad).to(fm.device)).cpu().numpy() * dtm_scale
                    else:
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
    dmin_draw = _dmin_tracker(draw_bank) if draw_bank is not None else None
    # per-bank root-anchored candidate sets (Kaveh: 'my triangle inequality idea -- add it
    # to the loss side, and the draw side'): one full-bank row at the root per move, leaves
    # query the nearest-256 (+tail since anchor); exact-vs-pruned audits printed per bank.
    cands = {n: {"idx": None, "ver": 0, "cache": {}, "audit": 0}
             for n in ("win", "loss", "draw")}

    def _anchor_bank(name, bk, F):
        c = cands[name]
        c["cache"] = {}
        if bk is None or len(bk) < 256:
            c["idx"] = None; return
        d_all = _bank_row(F, bk.embs)
        c["idx"] = np.argsort(d_all)[:128]; c["ver"] = len(bk); c["audit"] += 1

    def _dmin_pruned(name, bk, tracker, keys):
        """min-distance via the root candidate annulus; falls back to the exact tracker
        when un-anchored. Move-local cache only (approx values never persist)."""
        c = cands[name]
        if bk is None or len(bk) == 0:
            return None
        if c["idx"] is None:
            return tracker(keys)
        cc = c["cache"]
        miss = [i for i, k in enumerate(keys) if k not in cc]
        if miss:
            tt = time.perf_counter()
            F = np.stack([emb_cache[keys[i]] for i in miss])
            sub = bk.embs[c["idx"]]
            if len(bk) > c["ver"]:
                sub = np.concatenate([sub, bk.embs[c["ver"]:]])
            d_sub = fm.d_to_bank(F, sub)
            if c["audit"] % 25 == 1 and not cc:
                d_ex = fm.d_to_bank(F, bk.embs)
                print(f"    [prune-audit:{name}] max|err| {np.abs(d_sub - d_ex).max():.3f} "
                      f"cands {len(sub)}/{len(bk)}", flush=True)
            for i, dv in zip(miss, d_sub):
                cc[keys[i]] = float(dv)
            if times is not None:
                times["dbank_s"] = times.get("dbank_s", 0.0) + time.perf_counter() - tt
                times["dbank_n"] = times.get("dbank_n", 0) + len(miss)
        return np.array([cc[k] for k in keys])

    def _kappa(boards, keys):
        """STATE-DEPENDENT draw mass (Kaveh: the draw surfaces MOVE with the trajectory --
        but they move by RULES arithmetic, so the sufficient statistic is exact): base +
        clock deadline pressure + repetition proximity vs the live game history + banked
        stalemate proximity (the fixed surface component). Emergent behavior: when winning,
        rising kappa deflates v -> the engine spontaneously prefers zeroing moves (captures
        and pawn pushes reset the clock); when losing it SEEKS the surface (swindle mode)."""
        k = np.full(len(boards), KAPPA)
        clk = np.array([bd.halfmove_clock for bd in boards], dtype=float)
        k = k + np.exp((clk - 100.0) / 12.0)
        if game_ctx is not None and game_ctx.get("hist"):
            h = game_ctx["hist"]
            k = k + np.array([0.6 if h.get(kk, 0) >= 2 else (0.15 if h.get(kk, 0) == 1 else 0.0)
                              for kk in keys])
        if dmin_draw is not None and draw_bank is not None and len(draw_bank) > 0:
            d_s = _dmin_pruned("draw", draw_bank, dmin_draw, keys)
            k = k + 0.5 * np.exp(-d_s / max(float(np.median(d_s)), 1e-6))
        return k
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
        # probe_tri_carry verdict (2026-07-25): the bound is good for ~3 plies then falls
        # off a cliff (spearman 0.72@k5 -> 0.85@k1), while a full root row costs ~nothing
        # next to per-leaf subset work -- so anchor EVERY move and shrink C: k=1/C=128
        # beats k=5/C=256 on fidelity at half the leaf cost. No carry between moves.
        est = None
        if cand["idx"] is not None:
            sub = bank.embs[cand["idx"]]
            if len(bank) > cand["ver"]:
                sub = np.concatenate([sub, bank.embs[cand["ver"]:]])
            est = float(fm.d_to_bank(F, sub)[0])
        d_all = _bank_row(F, bank.embs)
        exact = float(d_all.min())
        if est is not None and (cand["audit"] % 10 == 1 or est - exact > 0.05):
            print(f"    [tri-refresh] dmin est {est:.3f} exact {exact:.3f} "
                  f"err {est - exact:+.3f}", flush=True)
        thr = exact + slack
        idx = np.flatnonzero(d_all <= thr)
        if len(idx) > 128 or len(idx) < 64:
            # dense-bank regime + measured carry curve: nearest-128 at every-move anchoring
            # (k=1/C=128: spearman 0.85, p95 0.03 plies) -- audits police it
            idx = np.argsort(d_all)[:128]
        cand["idx"] = idx; cand["ver"] = len(bank); cand["audit"] += 1
        _anchor_bank("loss", loss_bank, F)      # pruning symmetry: loss + draw banks get
        _anchor_bank("draw", draw_bank, F)      # the same root-anchored candidate scheme

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
            # whole batch in the nucleus: DTM path for p_w; kappa still needs embeds
            # when the draw bank is live
            d_w = _dtm_hat(boards, keys)
            p_w = np.exp(-d_w / dtm_scale)
            p_l = 0.0
            if nl or (draw_bank is not None and len(draw_bank) > 0):
                _embed(boards, keys)
            if nl:
                d_l = _dmin_pruned("loss", loss_bank, dmin_loss, keys)
                recent_l.extend(d_l.tolist())
                M_l = max(float(np.median(recent_l)), 1e-6)
                p_l = np.exp(-d_l / M_l)
            kap = _kappa(boards, keys)
            return (p_w - p_l) / (p_w + p_l + kap)
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
        d_l = _dmin_pruned("loss", loss_bank, dmin_loss, keys) if nl else None
        recent.extend((d_w if d_w is not None else d_l).tolist())
        M = max(float(np.median(recent)), 1e-6)
        p_w = np.exp(-d_w / M) if d_w is not None else 0.0
        if d_l is not None:
            recent_l.extend(np.asarray(d_l).tolist())
            M_l = max(float(np.median(recent_l)), 1e-6)
            p_l = np.exp(-np.asarray(d_l) / M_l)
        else:
            p_l = 0.0
        kap = _kappa(boards, keys)
        v = (p_w - p_l) / (p_w + p_l + kap)
        if nuc:                                     # mixed batch: nucleus rows -> DTM value
            db = [boards[i] for i in nuc]; dk = [keys[i] for i in nuc]
            pw_n = np.exp(-_dtm_hat(db, dk) / dtm_scale)
            pl_n = (p_l[nuc] if isinstance(p_l, np.ndarray) else np.zeros(len(nuc)))
            v[nuc] = (pw_n - pl_n) / (pw_n + pl_n + kap[nuc])
        return v
    def diag(board):
        """Single-board CALIBRATED belief readout for UIs/probes. Raw bank distances are
        meaningless out-of-support (an opening reads d_loss~5 to an endgame mate under
        mc2); what the engine BELIEVES runs through the running-median temperatures --
        expose exactly that: v, p_w, p_l, kappa, plus the raws and scales for honesty."""
        boards = [board]; keys = [board.epd()]
        nw = len(bank); nl = len(loss_bank) if loss_bank is not None else 0
        in_nuc = dtm_net is not None and len(board.piece_map()) <= nucleus_max_pieces
        _embed(boards, keys)
        d_w = float(_dtm_hat(boards, keys)[0]) if in_nuc else \
            (float(dmin_win(keys)[0]) if nw else None)
        d_l = float(dmin_loss(keys)[0]) if nl else None
        M = max(float(np.median(recent)), 1e-6) if len(recent) else None
        M_l = max(float(np.median(recent_l)), 1e-6) if len(recent_l) else None
        scale_w = dtm_scale if in_nuc else M
        p_w = float(np.exp(-d_w / scale_w)) if (d_w is not None and scale_w) else 0.0
        p_l = float(np.exp(-d_l / M_l)) if (d_l is not None and M_l) else 0.0
        kap = float(_kappa(boards, keys)[0])
        v = (p_w - p_l) / (p_w + p_l + kap)
        return dict(v=round(v, 3), p_w=round(p_w, 3), p_l=round(p_l, 3),
                    kappa=round(kap, 3), d_win=d_w, d_loss=d_l,
                    M=M, M_l=M_l, nucleus=in_nuc)
    value_fn.diag = diag
    value_fn.set_anchor = set_anchor
    value_fn.invalidate_anchor = lambda: cand.update(idx=None)   # irreversible move played:
    return value_fn                                              # the +-1-ply bound is void


def make_planner(fm: FieldModel, bank: OnlineMateBank, give_up_plies: int = 20,
                 loss_bank=None, draw_bank=None, game_ctx=None, prior_fn=None,
                 rl_epsilon: float = 0.1):
    """THE HIERARCHICAL PLANNER (v2; Kaveh: 'from a seven piece, find the path to a
    winning five piece'). Discrete GOAL-REGION selection + execution via the prior
    alpha-dial (subgoals never contaminate the value). RULES + own-experience only:
      - goal vocabulary = material classes REACHABLE by an available trade (rules)
      - goal score = bank DENSITY in the class (mates we have actually achieved there --
        'winning 5-piece' is defined by where our own mates live, Kaveh's
        forceability x reachability x density prior) x field REACHABILITY
        exp(-d(F(x), class exemplars))
      - execution = bias captures transitioning toward the goal class (target piece =
        the class difference); handoff to direct once <= 6 pieces; give-up + reselect
        after give_up_plies without a class transition.
    Deterministic v1 selector; the PlanSelector protocol is the RL-swappable seam."""
    state = {"plan": "direct", "goal": None, "since": -999, "target_pt": None}

    # RL PLANSELECTOR hook (Kaveh: planner RL in the loop): when a trained ckpt exists,
    # plan TYPE is chosen by expected outcome over a cheap observation (epsilon-greedy);
    # the deterministic rules stay as fallback and the goal machinery is unchanged.
    rl = {"net": None, "path": None, "rng": np.random.default_rng(0)}

    def _rl_load():
        import glob as _g, re as _re
        cands = _g.glob("data/derived/sep/planner_rl_r*.pt")
        if not cands:
            return
        best = max(cands, key=lambda p: int(_re.search(r"r(\d+)\.pt", p).group(1)))
        if best != rl["path"]:
            import torch
            from experiments.train_planner_rl import PlanNet, PLANS, OBS_KEYS, featurize
            st = torch.load(best, map_location="cpu", weights_only=False)
            net = PlanNet(); net.load_state_dict(st["state"]); net.eval()
            rl.update(net=net, path=best, mu=st["mu"], sd=st["sd"],
                      plans=st["plans"], featurize=featurize)
            print(f"[planner] RL selector loaded: {best}", flush=True)

    def _cheap_obs(b: chess.Board) -> dict:
        o = {"clock": b.halfmove_clock, "clock_headroom": 100 - b.halfmove_clock,
             "n_pieces": len(b.piece_map()),
             "rep_max_nearby": (game_ctx or {}).get("hist", {}).get(b.epd(), 0),
             "seen_in_game": (game_ctx or {}).get("hist", {}).get(b.epd(), 0),
             "n_win": len(bank), "n_loss": len(loss_bank) if loss_bank else 0,
             "n_draw": len(draw_bank) if draw_bank else 0}
        try:
            F = fm.embed_F_boards([b])
            if len(bank):
                o["d_win"] = float(fm.d_to_bank(F, bank.embs)[0])
                o["class_density"] = int(len(bank.class_idx(mat_sig(b))))
            if loss_bank is not None and len(loss_bank):
                o["d_loss"] = float(fm.d_to_bank(F, loss_bank.embs)[0])
            if draw_bank is not None and len(draw_bank):
                o["d_draw"] = float(fm.d_to_bank(F, draw_bank.embs)[0])
            if prior_fn is not None:
                pri = np.array(list(prior_fn(b).values()))
                pri = pri[pri > 0]
                o["prior_entropy"] = float(-(pri * np.log(pri)).sum())
                o["prior_top1"] = float(pri.max())
        except Exception:
            pass
        return o

    def _rl_pick(b: chess.Board) -> str | None:
        _rl_load()
        if rl["net"] is None or rl["rng"].random() < rl_epsilon:
            return None                              # explore / no model: rules decide
        import torch
        x = (rl["featurize"](_cheap_obs(b)) - rl["mu"]) / rl["sd"]
        oh = np.eye(len(rl["plans"]), dtype=np.float32)
        with torch.no_grad():
            scores = rl["net"](torch.from_numpy(np.tile(x, (len(rl["plans"]), 1)).astype(np.float32)),
                               torch.from_numpy(oh)).numpy()
        return rl["plans"][int(np.argmax(scores))]

    def plan(b: chess.Board, plies: int) -> dict:
        n = len(b.piece_map())
        choice = _rl_pick(b)
        if choice == "reset" or (choice is None and b.halfmove_clock >= 30 and n <= 6):
            state.update(plan="reset", goal=None, target_pt=None)
            return state
        if n <= 6:
            if choice is None and b.halfmove_clock >= 30:
                state.update(plan="reset", goal=None, target_pt=None)
            else:
                state.update(plan="direct", goal=None, target_pt=None)
            return state
        cur = mat_sig(b)
        if state["goal"] is None or cur == state["goal"] or plies - state["since"] >= give_up_plies:
            cands: dict[str, int] = {}
            for m in b.legal_moves:
                if b.is_capture(m):
                    c = b.copy(stack=False); c.push(m)
                    cands[mat_sig(c)] = c.piece_type_at(m.to_square) or 0  # mover's pt, unused
            if not cands:
                state.update(plan="direct", goal=None, target_pt=None)
                return state
            F = fm.embed_F_boards([b])
            best_sig, best_s = None, -1.0
            for sig in cands:
                idx = bank.class_idx(sig)
                density = float(len(idx))
                if len(idx) > 0:
                    reach = float(np.exp(-fm.d_to_bank(F, bank.embs[idx[:256]]).min() / 20.0))
                else:
                    reach = 0.5                      # unexplored class: neutral reachability
                s = (density + 1.0) * reach          # +1 smoothing: unexplored != impossible
                if s > best_s:
                    best_sig, best_s = sig, s
            state.update(plan="tradedown", goal=best_sig, since=plies)
            # the class difference names the piece TYPE whose capture makes the transition
            state["target_pt"] = None
            for m in b.legal_moves:
                if b.is_capture(m):
                    c = b.copy(stack=False); c.push(m)
                    if mat_sig(c) == best_sig:
                        state["target_pt"] = (chess.PAWN if b.is_en_passant(m)
                                              else b.piece_type_at(m.to_square))
                        break
        return state
    return plan


def make_batched_energy_prior(ckpt: str, cohort: int = 11, device: str = "cpu",
                              times: dict | None = None, game_ctx: dict | None = None,
                              plan_alpha: float = 1.0):
    """(policy_fn, policy_batch_fn) sharing one net + one epd cache. The batch fn is the
    speed path: one forward for a whole leaf batch instead of ~2ms singles (mcts.py already
    supports policy_batch_fn -- no search-semantics change, same net, same numbers)."""
    import torch
    from catspace.research.tools.chess_specific.chessdata.encode import encode_meta, encode_packed
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import pick_device
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.opponent import OpponentModel
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

    def _bias(b, pri):
        """plan alpha-dial: multiplicative prior reweighting toward plan-aligned moves.
        reset -> zeroing moves; tradedown -> captures of the GOAL-class target piece
        (goal-directed, not 'all captures'). Cache stays BASE priors; bias at retrieval."""
        plan = game_ctx.get("plan") if game_ctx is not None else None
        if plan not in ("reset", "tradedown") or plan_alpha <= 0:
            return pri
        boost = float(np.exp(plan_alpha))
        tpt = game_ctx.get("target_pt")

        def aligned(m):
            if plan == "reset":
                return b.is_capture(m) or b.piece_type_at(m.from_square) == chess.PAWN
            if tpt is None:
                return b.is_capture(m)
            cap_pt = chess.PAWN if b.is_en_passant(m) else b.piece_type_at(m.to_square)
            return b.is_capture(m) and cap_pt == tpt
        out = {m: (p * boost if aligned(m) else p) for m, p in pri.items()}
        z = sum(out.values())
        return {m: p / z for m, p in out.items()} if z > 0 else pri

    def policy_fn(b):
        k = b.epd()
        if k not in cache:
            cache[k] = _forward([b])[0]
        return _bias(b, cache[k])

    def policy_batch_fn(boards):
        keys = [b.epd() for b in boards]
        miss_i = [i for i, k in enumerate(keys) if k not in cache]
        if miss_i:
            for i, pri in zip(miss_i, _forward([boards[i] for i in miss_i])):
                cache[keys[i]] = pri
        return [_bias(b, cache[k]) for b, k in zip(boards, keys)]

    return policy_fn, policy_batch_fn


def tb_white_move(b, tb):
    """Standard DTZ-conversion move (g017 autopsy: min-|dtz-after| has NO gradient when a
    zeroing move is reachable from everywhere -- every rook shuffle scored dtz=2 and the
    engine waltzed 50 moves without progress; tb.py's documented DTZ!=DTM trap):
      1. winning ZEROING move (capture/pawn push) if one exists -- immediate progress, and
         inherently claim-safe (new material voids all repetition history);
      2. else winning move with STRICTLY decreasing |dtz|, claim-safe first;
      3. else any claim-safe winning move; else best-dtz winning move.
    Consulted only on fallback triggers; every use is LOGGED (attribution)."""
    _, dz_now = tb.wdl_dtz(b)
    dz_now = abs(dz_now) if dz_now is not None else 999
    zeroing, progress, others = [], [], []
    for m in b.legal_moves:
        c = b.copy(stack=False); c.push(m)
        w, dz = tb.wdl_dtz(c)
        if w is None or -w != 2:
            continue
        dza = abs(dz) if dz is not None else 999
        if b.is_capture(m) or b.piece_type_at(m.from_square) == chess.PAWN:
            zeroing.append((dza, m))
        elif dza < dz_now:
            progress.append((dza, m))
        else:
            others.append((dza, m))
    if zeroing:
        return min(zeroing, key=lambda x: x[0])[1]
    for pool in (progress, others):
        for _, m in sorted(pool, key=lambda x: x[0]):
            b.push(m)
            ok = not b.can_claim_threefold_repetition()
            b.pop()
            if ok:
                return m
    all_ = progress + others
    return min(all_, key=lambda x: x[0])[1] if all_ else None
