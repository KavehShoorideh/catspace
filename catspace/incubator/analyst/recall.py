"""recall.py -- MEMORY INDEXED BY CONCEPTS (Kaveh 2026-09-12): "sample random positions from human games, store the
state of the concepts in a DB indexed by the concept values; for an unknown position ask what I remember that is
closest, and what I remember that is in its FUTURE -- reachable from here -- by a quasimetric rather than by
rules, so it generalises; then which of those futures I want and which I must avoid."

The memory: every position we hold a graph for (the random Lichess positions of ChessBench with the teacher's
state value; Kaveh's games with the result and the engine's evaluation), each stored with its geometry coordinates
psi (the quasimetric space the knowledge network learned from the games' trajectories), its Backpack sum phi, its
regime code, its value from the mover's view, its piece counts and its FEN. The index is the geometry: the nearest
memories are the smallest d(query -> m) + d(m -> query); the FUTURES are the memories with the smallest d(query ->
m) -- the direction of time is what the quasimetric learned. The rules' answer to 'reachable' (no piece type's
count rises, pawns never rise, castling rights never return) is computed beside it as the ground truth the
geometry is measured against, never as the filter. The CONTRAST: among the reachable futures, the highest-valued
are where to steer, the lowest where not to go.

Gauges: (1) reachability -- do a game's actual later positions rank nearer than random memories, by the geometry
and by the rules; (2) value by recall -- the k nearest memories' mean value against the teacher's, beside the value
head; (3) recall as a move picker -- each legal move's child scored by its nearest reachable memories' value.

    python -m catspace.incubator.analyst.recall build --ckpt data/derived/checkpoints/kn_m256/best.pt
    python -m catspace.incubator.analyst.recall gauge
    python -m catspace.incubator.analyst.recall ask --fen "<fen>"
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np
import torch

from .memo import Memo
from .graph import build
from . import knowledge_net as K
from .regimes import Codebook
from .experts import net_sample

STORE = Path("data/derived/recall")
DEFAULT_CKPT = "data/derived/checkpoints/kn_m256/best.pt"
DATA = "data/derived/knowledge,data/derived/chessbench_kn"


def counts_of(board: chess.Board) -> np.ndarray:
    """(12,) piece counts: our P N B R Q K then theirs, from the side to move's view."""
    us = board.turn
    return np.array([len(board.pieces(pt, c)) for c in (us, not us) for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)], dtype=np.int8)


def castle_bits(board: chess.Board) -> int:
    """Castling rights as four bits (K Q k q), small enough to store."""
    return (board.has_kingside_castling_rights(chess.WHITE) << 0) | (board.has_queenside_castling_rights(chess.WHITE) << 1) | \
           (board.has_kingside_castling_rights(chess.BLACK) << 2) | (board.has_queenside_castling_rights(chess.BLACK) << 3)


def reachable_by_rules(now: np.ndarray, mem: np.ndarray, now_castle: int, mem_castle: int, same_side: bool) -> bool:
    """A necessary condition for the memory to lie in the future of the present position: no piece type's count
    rises except by promotion (a queen, rook, bishop or knight may rise only if the pawns fell by at least as
    much), castling rights never come back. Sides are matched by who is to move (memories with the other side to
    move are compared with the counts swapped)."""
    m = mem if same_side else np.concatenate([mem[6:], mem[:6]])
    for s in (0, 6):
        pawns_lost = int(now[s]) - int(m[s])
        if pawns_lost < 0:
            return False
        promoted = sum(max(0, int(m[s + t]) - int(now[s + t])) for t in (1, 2, 3, 4))
        if promoted > pawns_lost:
            return False
        if m[s + 5] > now[s + 5]:
            return False
    return (mem_castle & ~now_castle) == 0


@dataclass
class Memory:
    psi: np.ndarray            # (N, d_geo)
    phi: np.ndarray            # (N, d)
    code: np.ndarray           # (N,)
    value: np.ndarray          # (N,) the mover's win probability (the teacher's state value; else the result; else tanh cp mapped)
    counts: np.ndarray         # (N, 12)
    castle: np.ndarray         # (N,)
    side: np.ndarray           # (N,) bool
    fen: list
    game: np.ndarray           # (N,)
    ply: np.ndarray            # (N,)
    source: np.ndarray         # (N,) 'cb' | 'games'

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / "memory.npz", psi=self.psi, phi=self.phi, code=self.code, value=self.value, counts=self.counts, castle=self.castle, side=self.side, game=self.game, ply=self.ply, source=self.source)
        json.dump(self.fen, open(path / "fens.json", "w"))

    @classmethod
    def load(cls, path: Path = STORE) -> "Memory":
        z = np.load(path / "memory.npz", allow_pickle=False)
        return cls(z["psi"], z["phi"], z["code"], z["value"], z["counts"], z["castle"], z["side"], json.load(open(path / "fens.json")), z["game"], z["ply"], z["source"])

    def __len__(self):
        return len(self.psi)


class Recall:
    def __init__(self, ckpt: str = DEFAULT_CKPT, store: Path = STORE, data_dir: str = "data/derived/knowledge"):
        self.model, self.cfg, self.device = K.load_model(ckpt)
        meta = json.load(open(Path(ckpt).parent / "meta.json"))
        self.buckets = K.Buckets.from_state(self.cfg.buckets, meta["buckets"]); self.pos_buckets = K.Buckets.from_state(self.cfg.buckets, meta["pos_buckets"])
        self.codebook = Codebook.load(str(Path(data_dir) / "codebook.npz"))
        self.mem = Memory.load(store) if (store / "memory.npz").exists() else None
        self._psi_t = torch.as_tensor(self.mem.psi, device=self.device) if self.mem is not None else None

    @torch.no_grad()
    def embed(self, board: chess.Board, g=None, memo=None):
        g = g if g is not None else build(board, memo if memo is not None else Memo())
        s, _ = net_sample(g, board, self.buckets, self.pos_buckets)
        b = K.collate([s], self.device); o = self.model(b)
        phi = o["phi"]; psi = self.model.psi(phi)
        return phi[0].cpu().numpy(), psi[0].cpu().numpy(), int(self.codebook.assign(g.position_vector()[None])[0]), g

    @torch.no_grad()
    def distances(self, psi_q: np.ndarray):
        """d(query -> every memory) and d(every memory -> query), in the geometry's plies."""
        q = torch.as_tensor(psi_q, device=self.device)[None]
        fwd = self.model.iqe.pairwise(q, self._psi_t)[0]; back = self.model.iqe.pairwise(self._psi_t, q)[:, 0]
        return fwd.cpu().numpy(), back.cpu().numpy()

    def ask(self, board: chess.Board, k: int = 8, horizon: float = 12.0, exclude_game=None, memo=None) -> dict:
        phi, psi, code, g = self.embed(board, memo=memo)
        fwd, back = self.distances(psi)
        now_c = counts_of(board); now_castle = castle_bits(board)
        keep = np.ones(len(self.mem), dtype=bool)
        if exclude_game is not None:
            keep &= self.mem.game != exclude_game
        sym = fwd + back
        nearest = [int(i) for i in np.argsort(np.where(keep, sym, np.inf))[:k]]
        # futures: nearest by the forward distance, the rules' verdict beside each
        order = np.argsort(np.where(keep, fwd, np.inf))
        futures = []
        for i in order[: 40 * k]:
            i = int(i)
            if fwd[i] > horizon and len(futures) >= k:
                break
            ok = reachable_by_rules(now_c, self.mem.counts[i], now_castle, int(self.mem.castle[i]), bool(self.mem.side[i]) == board.turn)
            futures.append(dict(i=i, d=float(fwd[i]), back=float(back[i]), rules=ok, value=float(self.mem.value[i]) if bool(self.mem.side[i]) == board.turn else 1.0 - float(self.mem.value[i]),
                                code=int(self.mem.code[i]), fen=self.mem.fen[i]))
            if len(futures) >= 3 * k:
                break
        reach = [f for f in futures if f["rules"] and f["fen"] != board.fen()]
        by_v = sorted(reach, key=lambda f: -f["value"])
        def distinct(fs):
            seen, out = set(), []
            for f in fs:
                t = self.codebook.describe(f["code"])
                if t not in seen:
                    seen.add(t); out.append(f)
            return out
        mid = float(np.median([f["value"] for f in reach])) if reach else 0.5
        steer = distinct([f for f in by_v if f["value"] >= mid + 0.08])[:3]
        avoid = distinct([f for f in by_v[::-1] if f["value"] <= mid - 0.08])[:3]
        return dict(phi=phi, psi=psi, code=code, nearest=nearest, futures=futures[:k], steer=steer, avoid=avoid, graph=g)

    def lines(self, board: chess.Board, ans: dict, k: int = 3) -> list[str]:
        cb = self.codebook; who = "White" if board.turn else "Black"
        def grouped(items, key_v, key_d=None):
            """Memories with the same words are one memory to the reader: mean value, count, nearest distance."""
            g = {}
            for it in items:
                t = cb.describe(it["code"]); e = g.setdefault(t, dict(n=0, v=0.0, d=math.inf, rules=True))
                e["n"] += 1; e["v"] += key_v(it); e["d"] = min(e["d"], key_d(it) if key_d else 0.0); e["rules"] = e["rules"] and it.get("rules", True)
            return sorted(((t, e["v"] / e["n"], e["n"], e["d"], e["rules"]) for t, e in g.items()), key=lambda x: -x[2])
        near = [dict(code=int(self.mem.code[i]), value=float(self.mem.value[i]) if bool(self.mem.side[i]) == board.turn else 1.0 - float(self.mem.value[i])) for i in ans["nearest"]]
        out = ["What I remember nearest: " + "; ".join(f"{t} (value {v:.2f}{f', x{n}' if n > 1 else ''})" for t, v, n, _, _ in grouped(near, lambda x: x["value"])[:k]) + "."]
        if ans["futures"]:
            out.append("In this position's future, by the geometry: " + "; ".join(f"{t} ({d:.0f} plies{'' if r else ', the rules say no'}, value {v:.2f}{f', x{n}' if n > 1 else ''})" for t, v, n, d, r in grouped(ans["futures"], lambda x: x["value"], lambda x: x["d"])[:k]) + ".")
        if ans["steer"]:
            out.append(f"Worth steering to (for {who}): " + "; ".join(f"{cb.describe(f['code'])} (value {f['value']:.2f}, {f['d']:.0f} plies)" for f in ans["steer"][:2]) + ".")
        if ans["avoid"]:
            out.append("To avoid: " + "; ".join(f"{cb.describe(f['code'])} (value {f['value']:.2f}, {f['d']:.0f} plies)" for f in ans["avoid"][:2]) + ".")
        return out


# ---------------------------------------------------------------- building the memory from the datasets
@torch.no_grad()
def build_memory(ckpt: str = DEFAULT_CKPT, data: str = DATA, out: Path = STORE, batch: int = 512, log=print):
    model, cfg, device = K.load_model(ckpt)
    meta = json.load(open(Path(ckpt).parent / "meta.json"))
    ds = K.Data(data, buckets=K.Buckets.from_state(cfg.buckets, meta["buckets"]), pos_buckets=K.Buckets.from_state(cfg.buckets, meta["pos_buckets"]), n_buckets=cfg.buckets, n_codes=cfg.n_codes, log=lambda *a, **k: None)
    N = len(ds.meta); t0 = time.perf_counter()
    psis, phis = [], []
    for s in range(0, N, batch):
        idx = np.arange(s, min(N, s + batch)); b = ds.batch(idx, device); o = model(b)
        phis.append(o["phi"].cpu().numpy()); psis.append(model.psi(o["phi"]).cpu().numpy())
        if (s // batch) % 40 == 0:
            log(f"[recall] embedded {s}/{N}, {time.perf_counter() - t0:.0f}s", flush=True)
    phi = np.concatenate(phis); psi = np.concatenate(psis)
    value = np.zeros(N, np.float32); source = np.empty(N, dtype="<U5"); counts = np.zeros((N, 12), np.int8); castle = np.zeros(N, np.int64); side = np.zeros(N, bool)
    fens = []
    for i, m in enumerate(ds.meta):
        board = chess.Board(m["fen"]); fens.append(m["fen"]); counts[i] = counts_of(board); castle[i] = castle_bits(board); side[i] = board.turn
        if m.get("wp") is not None:
            value[i] = m["wp"]; source[i] = "cb"                                             # the teacher's state value
        else:
            value[i] = 0.5 + 0.5 * math.tanh((m.get("best_cp") or 0) / 400.0); source[i] = "games"   # the engine's evaluation, not the game's result (too noisy a value for one position)
    mem = Memory(psi.astype(np.float32), phi.astype(np.float32), ds.code_now.astype(np.int16), value, counts, castle, side, fens, ds.game.astype(np.int64), ds.ply.astype(np.int64), source)
    mem.save(out)
    log(f"[recall] memory of {N} positions ({(source == 'cb').sum()} random Lichess positions with the teacher's value, {(source == 'games').sum()} game positions) in {time.perf_counter() - t0:.0f}s -> {out}", flush=True)
    return mem, ds


# ---------------------------------------------------------------- gauges
def gauge(ckpt: str = DEFAULT_CKPT, n: int = 1500, k: int = 16, seed: int = 0, log=print):
    from sklearn.metrics import roc_auc_score
    rc = Recall(ckpt); mem = rc.mem
    ds = K.Data(DATA, buckets=rc.buckets, pos_buckets=rc.pos_buckets, n_buckets=rc.cfg.buckets, n_codes=rc.cfg.n_codes, log=lambda *a, **kw: None)
    # the memory rows are the dataset rows in order; queries come from the test split of the games (they have futures)
    rng = random.Random(seed)
    test_games = [int(i) for i in ds.ids("test") if ds.meta[int(i)].get("site") != "chessbench" and ds.next[int(i)] >= 0]
    rng.shuffle(test_games); queries = test_games[:n]
    aucs_geo, aucs_rules, arrow = [], [], []
    v_recall, v_head, v_true = [], [], []
    t0 = time.perf_counter()
    for qi, i in enumerate(queries):
        m = ds.meta[i]; board = chess.Board(m["fen"])
        psi = mem.psi[i]; fwd, back = rc.distances(psi)
        g = int(ds.game[i]); same = mem.game == g
        later = same & (mem.ply > ds.ply[i]) & (mem.ply <= ds.ply[i] + 20)
        other = ~same
        if later.sum() < 3:
            continue
        # reachability: true later positions of the game against random other memories, scored by -d(query -> m)
        neg = np.where(other)[0]; neg = rng.sample(list(neg), min(200, len(neg)))
        pos = np.where(later)[0]
        y = np.array([1] * len(pos) + [0] * len(neg)); sc = -np.concatenate([fwd[pos], fwd[neg]])
        aucs_geo.append(roc_auc_score(y, sc))
        now_c = counts_of(board); nc = castle_bits(board)
        rules = np.array([reachable_by_rules(now_c, mem.counts[j], nc, int(mem.castle[j]), bool(mem.side[j]) == board.turn) for j in np.concatenate([pos, neg])], dtype=float)
        aucs_rules.append(roc_auc_score(y, rules))
        arrow.append(float((fwd[pos] < back[pos]).mean()))
        # value by recall: the k nearest OTHER-game memories (symmetric distance) with the mover's view matched
        sym = fwd + back; sym[same] = np.inf
        nn = np.argsort(sym)[:k]
        vals = [float(mem.value[j]) if bool(mem.side[j]) == board.turn else 1.0 - float(mem.value[j]) for j in nn]
        v_recall.append(np.mean(vals))
        v_true.append(0.5 + 0.5 * math.tanh((m.get("best_cp") or 0) / 400.0))
        with torch.no_grad():
            b = ds.batch([i], rc.device); v_head.append(0.5 + 0.5 * float(torch.tanh(rc.model.cp_head(rc.model(b)["phi"])[0, 0])))
        if (qi + 1) % 200 == 0:
            log(f"[recall gauge] {qi + 1}/{len(queries)} queries, {time.perf_counter() - t0:.0f}s", flush=True)
    vt, vr, vh = np.array(v_true), np.array(v_recall), np.array(v_head)
    r2 = lambda p: 1 - ((p - vt) ** 2).mean() / vt.var()
    out = [f"[recall] memory {len(mem)} positions; {len(aucs_geo)} game queries with a future in memory",
           f"  REACHABILITY (a game's own later positions within 20 plies against 200 random memories): AUC by the geometry {np.mean(aucs_geo):.3f} | by the rules' exclusions {np.mean(aucs_rules):.3f} (0.5 = nothing)",
           f"  the arrow on those pairs: d(now -> later) < d(later -> now) {np.mean(arrow):.0%}",
           f"  VALUE by recall ({k} nearest other-game memories, mover's view) R2 {r2(vr):.3f} | the value head {r2(vh):.3f} | recall+head averaged {r2(0.5 * (vr + vh)):.3f}; corr recall-vs-true {np.corrcoef(vr, vt)[0, 1]:.3f}"]
    log("\n".join(out)); return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["build", "gauge", "ask"]); ap.add_argument("--ckpt", default=DEFAULT_CKPT); ap.add_argument("--fen", default="")
    ap.add_argument("--n", type=int, default=1500)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build_memory(a.ckpt)
    elif a.cmd == "gauge":
        gauge(a.ckpt, a.n)
    else:
        rc = Recall(a.ckpt); b = chess.Board(a.fen); ans = rc.ask(b)
        for l in rc.lines(b, ans):
            print(l)


if __name__ == "__main__":
    main()
