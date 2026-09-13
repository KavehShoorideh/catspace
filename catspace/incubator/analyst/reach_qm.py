"""reach_qm.py -- a SIMPLE reachability quasimetric conditioned on Elo (Kaveh 2026-09-13): d_elo(s -> s') in plies,
from the board alone, so it trains on millions of positions from games at every level.

Encoder: piece tokens (identity + square) plus an Elo token and a side token through a small transformer, mean
pooled, to psi in a 64-dimensional interval quasimetric space (Wang & Isola's IQE). Objective, from games:
forward pairs (s_t, s_{t+k}) of one game are calibrated to k plies; every pair not known reachable -- other
games, and the reverse of time within a game -- is pushed apart; reachable pairs are bounded by their ply gap;
VICReg's spread keeps the space from collapsing onto a ply clock. Nothing names pieces or pawns. The Elo of the
game is an input to both endpoints, so the distance is the distance AT THAT LEVEL: what a 1200 reaches in ten
plies is not what a 2200 does.

Gauges on held-out games, by rating band: calibration (predicted plies against the true gap), reachability (a
game's own later positions against random positions, AUC), the arrow (d forward < d backward), and whether the
Elo conditioning matters (the same pairs scored at the wrong Elo).

    python -m catspace.incubator.analyst.reach_qm prepare --pgn data/external/lichess/lichess_2015-06.pgn.zst --games 60000
    python -m catspace.incubator.analyst.reach_qm train --epochs 6
    python -m catspace.incubator.analyst.reach_qm gauge
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import time
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from catspace.research.components.encoder.approaches.jepa_tokenizer.src.iqe import IQE

OUT = Path("data/derived/reach_qm")
ELO_BANDS = ((0, 1200), (1200, 1500), (1500, 1800), (1800, 2100), (2100, 4000))


# ---------------------------------------------------------------- data: games with Elo, positions as piece lists
def prepare(pgn_path: str, n_games: int = 60000, min_plies: int = 20, log=print):
    import zstandard
    OUT.mkdir(parents=True, exist_ok=True)
    games = []; t0 = time.perf_counter(); seen = 0
    with open(pgn_path, "rb") as fh:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8", errors="ignore")
        while len(games) < n_games:
            g = chess.pgn.read_game(text)
            if g is None:
                break
            seen += 1
            h = g.headers
            try:
                we, be = int(h.get("WhiteElo", 0)), int(h.get("BlackElo", 0))
            except ValueError:
                continue
            if we <= 0 or be <= 0 or h.get("Variant", "Standard") != "Standard":
                continue
            moves = [m.uci() for m in g.mainline_moves()]
            if len(moves) < min_plies:
                continue
            games.append(dict(elo_w=we, elo_b=be, result=h.get("Result", "*"), tc=h.get("TimeControl", ""), moves=moves))
            if len(games) % 5000 == 0:
                log(f"[reach_qm] {len(games)} games kept of {seen} read, {time.perf_counter() - t0:.0f}s", flush=True)
    with open(OUT / "games.jsonl", "w") as f:
        for g in games:
            f.write(json.dumps(g) + "\n")
    elos = np.array([(g["elo_w"] + g["elo_b"]) / 2 for g in games])
    log(f"[reach_qm] {len(games)} games, {sum(len(g['moves']) for g in games)} plies; mean Elo {elos.mean():.0f}, bands: " +
        ", ".join(f"{lo}-{hi}: {int(((elos >= lo) & (elos < hi)).sum())}" for lo, hi in ELO_BANDS), flush=True)


def encode_board(board: chess.Board):
    """(32,) piece type 1-12 from the side to move's view (0 = pad), (32,) squares mirrored so ranks read from the mover's side."""
    us = board.turn
    types = np.zeros(32, np.int64); sqs = np.full(32, 64, np.int64); i = 0
    for sq, p in board.piece_map().items():
        if i >= 32:
            break
        types[i] = p.piece_type + (0 if p.color == us else 6); sqs[i] = sq if us else (sq ^ 56); i += 1
    return types, sqs


class Positions:
    """All positions of the prepared games, tokenized once; pairs are drawn by game and ply."""
    def __init__(self, path: Path = OUT / "games.jsonl", max_games: int = 0, split_seed: int = 0, log=print):
        t0 = time.perf_counter()
        games = [json.loads(l) for l in open(path)]
        if max_games:
            games = games[:max_games]
        rng = random.Random(split_seed); order = list(range(len(games))); rng.shuffle(order)
        n = len(games); test = set(order[: n // 10]); val = set(order[n // 10: n // 5])
        types, sqs, game_id, ply, elo, castle = [], [], [], [], [], []
        self.split = []
        for gi, g in enumerate(games):
            b = chess.Board(); e = (g["elo_w"] + g["elo_b"]) / 2
            sp = "test" if gi in test else "val" if gi in val else "train"
            for p, u in enumerate(g["moves"]):
                t, s = encode_board(b); types.append(t); sqs.append(s); game_id.append(gi); ply.append(p); elo.append(e); castle.append(int(b.has_castling_rights(chess.WHITE)) + 2 * int(b.has_castling_rights(chess.BLACK)))
                self.split.append(sp)
                try:
                    b.push_uci(u)
                except Exception:
                    break
        self.types = np.stack(types); self.sqs = np.stack(sqs); self.game = np.array(game_id); self.ply = np.array(ply); self.elo = np.array(elo, np.float32); self.castle = np.array(castle)
        self.split = np.array(self.split)
        self.index = {}
        for i, (g, p) in enumerate(zip(self.game, self.ply)):
            self.index[(int(g), int(p))] = i
        self.by_game = {}
        for i, g in enumerate(self.game):
            self.by_game.setdefault(int(g), []).append(i)
        log(f"[reach_qm] {len(self.types)} positions from {len(games)} games in {time.perf_counter() - t0:.0f}s (train {(self.split == 'train').sum()}, val {(self.split == 'val').sum()}, test {(self.split == 'test').sum()})", flush=True)

    def ids(self, split):
        return np.nonzero(self.split == split)[0]

    def batch(self, idx, device, elo=None):
        idx = np.asarray(idx)
        e = self.elo[idx] if elo is None else np.full(len(idx), elo, np.float32)
        return dict(types=torch.as_tensor(self.types[idx]).to(device), sqs=torch.as_tensor(self.sqs[idx]).to(device),
                    elo=torch.as_tensor((e - 1500.0) / 400.0).to(device), game=torch.as_tensor(self.game[idx]).to(device), ply=torch.as_tensor(self.ply[idx]).to(device))


# ---------------------------------------------------------------- model
class ReachQM(nn.Module):
    def __init__(self, d: int = 64, layers: int = 3, heads: int = 4, d_geo: int = 64):
        super().__init__()
        self.type_emb = nn.Embedding(13, d); self.sq_emb = nn.Embedding(65, d)
        self.elo_proj = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)          # the nested fast path has no MPS kernel
        self.out = nn.Linear(d, d_geo)
        self.iqe = IQE(d_geo, components=16)

    def forward(self, b):
        x = self.type_emb(b["types"]) + self.sq_emb(b["sqs"])                            # (B, 32, d)
        elo = self.elo_proj(b["elo"][:, None])[:, None, :]                                 # (B, 1, d): the level as a token
        x = torch.cat([elo, x], 1)
        pad = torch.cat([torch.zeros(x.shape[0], 1, dtype=torch.bool, device=x.device), b["types"] == 0], 1)
        h = self.enc(x, src_key_padding_mask=pad)
        keep = (~pad).float()[..., None]
        return self.out((h * keep).sum(1) / keep.sum(1))                                    # (B, d_geo)


def losses(model, b, psi, n_pairs: int, k: torch.Tensor, margin: float = 40.0):
    """The first n_pairs rows are anchors, the next n_pairs their partners k plies ahead; the rest are extra
    positions. The quasimetric objective over the whole batch in both directions."""
    D = model.iqe.pairwise(psi, psi)
    games, plies = b["game"], b["ply"]
    same = games[:, None] == games[None, :]; later = same & (plies[None, :] > plies[:, None])
    gap = (plies[None, :] - plies[:, None]).float().clamp(min=1.0)
    eye = torch.eye(psi.shape[0], dtype=torch.bool, device=psi.device)
    pushed = (~later) & ~eye
    out = {}
    out["fwd"] = F.smooth_l1_loss(model.iqe(psi[:n_pairs], psi[n_pairs:2 * n_pairs]), k.float())
    out["neg"] = F.relu(margin - D)[pushed].mean() if pushed.any() else psi.sum() * 0
    out["bound"] = F.relu(D - gap)[later].mean() if later.any() else psi.sum() * 0
    z = psi - psi.mean(0, keepdim=True); std = torch.sqrt(z.var(0, unbiased=False) + 1e-4)
    cov = (z.T @ z) / max(1, psi.shape[0] - 1)
    out["vic"] = F.relu(1.0 - std).mean() + (cov - torch.diag(torch.diag(cov))).pow(2).sum() / psi.shape[1]
    out["total"] = out["fwd"] + out["neg"] + out["bound"] + 2.0 * out["vic"]
    return out


def train(epochs: int = 6, batch_pairs: int = 192, k_max: int = 40, lr: float = 3e-4, d: int = 64, layers: int = 3, max_games: int = 0, seed: int = 0, log=print):
    torch.manual_seed(seed); rng = random.Random(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    P = Positions(max_games=max_games, log=log)
    model = ReachQM(d=d, layers=layers).to(device)
    geo = [model.iqe.log_scale, model.iqe.alpha_logit]
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if all(p is not g for g in geo)]}, {"params": geo, "lr": lr * 30, "weight_decay": 0.0}], lr=lr, weight_decay=0.05)
    tr = [int(i) for i in P.ids("train")]
    # batches are drawn from ONE narrow rating window (200 points), so the Elo token cannot serve as a game
    # identity that separates a game's positions from the other games' in the batch for free
    windows = {}
    for i in tr:
        windows.setdefault(int(P.elo[i] // 200), []).append(i)
    wins = [w for w in windows.values() if len(w) >= 4 * batch_pairs]
    weights = np.array([len(w) for w in wins], dtype=float); weights /= weights.sum()
    steps_per_epoch = len(tr) // batch_pairs
    log(f"[reach_qm] model {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params; {steps_per_epoch} steps an epoch; {len(wins)} rating windows", flush=True)
    t0 = time.perf_counter(); step = 0
    OUT.mkdir(parents=True, exist_ok=True)
    for ep in range(epochs):
        acc = {}
        for s in range(steps_per_epoch):
            w = wins[int(np.random.default_rng(seed * 100003 + step).choice(len(wins), p=weights))]
            anchors = rng.sample(w, batch_pairs); a_ids, p_ids, ks = [], [], []
            for i in anchors:
                g, p = int(P.game[i]), int(P.ply[i]); kk = rng.randint(1, k_max); j = P.index.get((g, p + kk))
                if j is None:
                    continue
                a_ids.append(i); p_ids.append(j); ks.append(kk)
            if len(a_ids) < 8:
                continue
            b = P.batch(a_ids + p_ids, device)
            psi = model(b)
            L = losses(model, b, psi, len(a_ids), torch.tensor(ks, device=device))
            opt.zero_grad(set_to_none=True); L["total"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            for k_, v in L.items():
                acc[k_] = acc.get(k_, 0.0) + float(v)
            if step % 200 == 0:
                log(f"[reach_qm] step {step} " + " ".join(f"{k_} {v / 200:.3f}" for k_, v in acc.items()) + f" scale {math.exp(float(model.iqe.log_scale)):.2f} ({time.perf_counter() - t0:.0f}s)", flush=True); acc = {}
        torch.save(dict(model=model.state_dict(), d=d, layers=layers, epoch=ep), OUT / "model.pt")
        ev = gauge(model=model, P=P, device=device, n=600, log=lambda *a, **kw: None)
        log(f"[reach_qm] epoch {ep}: " + ev.split("\n")[1].strip() + " | " + ev.split("\n")[2].strip() + f" | {time.perf_counter() - t0:.0f}s", flush=True)
    return model, P


# ---------------------------------------------------------------- gauge
@torch.no_grad()
def gauge(model=None, P=None, device=None, n: int = 2000, seed: int = 1, log=print) -> str:
    from sklearn.metrics import roc_auc_score
    device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if model is None:
        ck = torch.load(OUT / "model.pt", map_location="cpu"); model = ReachQM(d=ck["d"], layers=ck["layers"]); model.load_state_dict(ck["model"]); model = model.to(device)
    model.eval()
    P = P or Positions(log=log)
    rng = random.Random(seed)
    test = [int(i) for i in P.ids("test")]; rng.shuffle(test)
    rows = []                                                       # (elo, k, d_fwd, d_back, auc, auc_wrong_elo)
    pool = [int(i) for i in P.ids("train")]
    for i in test[:n]:
        g, p = int(P.game[i]), int(P.ply[i]); ids = P.by_game[g]
        later = [j for j in ids if 0 < int(P.ply[j]) - p <= 40]
        if len(later) < 4:
            continue
        neg = rng.sample(pool, 100)
        rows_g = later + neg
        b = P.batch([i] + rows_g, device); psi = model(b)
        fwd = model.iqe.pairwise(psi[:1], psi[1:])[0].cpu().numpy(); back = model.iqe.pairwise(psi[1:], psi[:1])[:, 0].cpu().numpy()
        y = np.array([1] * len(later) + [0] * len(neg))
        auc = roc_auc_score(y, -fwd)
        # the same query and memories scored at the WRONG Elo (1500 away, clipped to the data's range)
        e = float(P.elo[i]); wrong = e + 800 if e < 1600 else e - 800
        bw = P.batch([i] + rows_g, device, elo=wrong); psw = model(bw)
        fw = model.iqe.pairwise(psw[:1], psw[1:])[0].cpu().numpy()
        auc_w = roc_auc_score(y, -fw)
        ks = np.array([int(P.ply[j]) - p for j in later])
        for kk, df, db in zip(ks, fwd[: len(later)], back[: len(later)]):
            rows.append((e, kk, float(df), float(db), auc, auc_w))
    r = np.array(rows)
    out = [f"[reach_qm] {len(set((row[0], row[4]) for row in rows))} held-out queries, {len(r)} forward pairs (k <= 40)"]
    calib = 1 - ((r[:, 2] - r[:, 1]) ** 2).mean() / r[:, 1].var()
    out.append(f"  calibration: predicted plies vs the true gap R2 {calib:.3f} (corr {np.corrcoef(r[:, 2], r[:, 1])[0, 1]:.3f}); mean |d - k| {np.abs(r[:, 2] - r[:, 1]).mean():.1f} plies")
    q = {}
    for row in rows:
        q.setdefault((row[0], row[4], row[5]), None)
    aucs = np.array([[e, a, aw] for (e, a, aw) in q])
    out.append(f"  reachability: a game's later positions against 100 random positions, AUC {aucs[:, 1].mean():.3f}; at the wrong Elo {aucs[:, 2].mean():.3f}; the arrow d(fwd) < d(back) {(r[:, 2] < r[:, 3]).mean():.0%}")
    out.append("  by rating band: " + "; ".join(f"{lo}-{hi}: n {int(((aucs[:, 0] >= lo) & (aucs[:, 0] < hi)).sum())}, AUC {aucs[((aucs[:, 0] >= lo) & (aucs[:, 0] < hi)), 1].mean():.3f}, arrow {(r[((r[:, 0] >= lo) & (r[:, 0] < hi)), 2] < r[((r[:, 0] >= lo) & (r[:, 0] < hi)), 3]).mean():.0%}"
                                                  for lo, hi in ELO_BANDS if ((aucs[:, 0] >= lo) & (aucs[:, 0] < hi)).sum() >= 5))
    # does the level change the distances? the same forward pairs scored at every band's centre
    model.train()
    text = "\n".join(out); log(text); return text


@torch.no_grad()
def gauge_hard(n: int = 1500, seed: int = 2, log=print) -> str:
    """Harder negatives for reachability: (a) the same game's EARLIER positions (the reverse of time), (b) other
    games' positions with the SAME piece counts as the query (material alone cannot tell them apart), (c) the same
    game's later positions beyond 40 plies (reachable but far: is the distance ordered?)."""
    from sklearn.metrics import roc_auc_score
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(OUT / "model.pt", map_location="cpu"); model = ReachQM(d=ck["d"], layers=ck["layers"]); model.load_state_dict(ck["model"]); model = model.to(device).eval()
    P = Positions(log=log)
    rng = random.Random(seed)
    test = [int(i) for i in P.ids("test")]; rng.shuffle(test)
    counts = np.stack([np.bincount(t, minlength=13)[1:] for t in P.types])                 # (N, 12) piece counts by type
    key = {}
    for i in P.ids("train"):
        key.setdefault(counts[i].tobytes(), []).append(int(i))
    a_earlier, a_same_material, spearman = [], [], []
    for i in test[:n]:
        g, p = int(P.game[i]), int(P.ply[i]); ids = P.by_game[g]
        later = [j for j in ids if 0 < int(P.ply[j]) - p <= 40]; earlier = [j for j in ids if 0 < p - int(P.ply[j]) <= 40]
        same_mat = key.get(counts[i].tobytes(), [])
        if len(later) < 4 or len(earlier) < 4:
            continue
        neg_m = rng.sample(same_mat, min(50, len(same_mat))) if len(same_mat) >= 5 else []
        rows = later + earlier + neg_m
        b = P.batch([i] + rows, device); psi = model(b)
        fwd = model.iqe.pairwise(psi[:1], psi[1:])[0].cpu().numpy()
        nl, ne = len(later), len(earlier)
        a_earlier.append(roc_auc_score([1] * nl + [0] * ne, -np.concatenate([fwd[:nl], fwd[nl:nl + ne]])))
        if neg_m:
            a_same_material.append(roc_auc_score([1] * nl + [0] * len(neg_m), -np.concatenate([fwd[:nl], fwd[nl + ne:]])))
        ks = np.array([int(P.ply[j]) - p for j in later])
        if len(ks) >= 6:
            from scipy.stats import spearmanr
            spearman.append(spearmanr(ks, fwd[:nl]).correlation)
    out = [f"[reach_qm hard] {len(a_earlier)} held-out queries",
           f"  later vs the same game's EARLIER positions (the reverse of time), AUC {np.mean(a_earlier):.3f}",
           f"  later vs other games' positions with the SAME piece counts, AUC {np.mean(a_same_material):.3f} ({len(a_same_material)} queries had such negatives)",
           f"  ordering within the future: Spearman(k, d) over a game's later positions {np.nanmean(spearman):.3f}"]
    text = "\n".join(out); log(text); return text


@torch.no_grad()
def gauge_elo(n: int = 1500, seed: int = 3, log=print) -> str:
    """Does the level change the distances? The same held-out forward pairs (k <= 40) scored with the Elo token set
    to each band's centre: the mean predicted plies per true gap, and the reverse-of-time AUC, at each level; plus
    each query at its own Elo against the others."""
    from sklearn.metrics import roc_auc_score
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(OUT / "model.pt", map_location="cpu"); model = ReachQM(d=ck["d"], layers=ck["layers"]); model.load_state_dict(ck["model"]); model = model.to(device).eval()
    P = Positions(log=log)
    rng = random.Random(seed)
    test = [int(i) for i in P.ids("test")]; rng.shuffle(test)
    levels = [1000, 1350, 1650, 1950, 2300]
    pred = {e: [] for e in levels}; aucs = {e: [] for e in levels}; own = []; gaps = []
    for i in test[:n]:
        g, p = int(P.game[i]), int(P.ply[i]); ids = P.by_game[g]
        later = [j for j in ids if 0 < int(P.ply[j]) - p <= 40]; earlier = [j for j in ids if 0 < p - int(P.ply[j]) <= 40]
        if len(later) < 4 or len(earlier) < 4:
            continue
        ks = np.array([int(P.ply[j]) - p for j in later]); gaps.append(ks)
        y = [1] * len(later) + [0] * len(earlier)
        for e in levels:
            b = P.batch([i] + later + earlier, device, elo=e); psi = model(b)
            fwd = model.iqe.pairwise(psi[:1], psi[1:])[0].cpu().numpy()
            pred[e].append(fwd[: len(later)]); aucs[e].append(roc_auc_score(y, -fwd))
        b = P.batch([i] + later + earlier, device); psi = model(b); fwd = model.iqe.pairwise(psi[:1], psi[1:])[0].cpu().numpy()
        own.append((float(P.elo[i]), roc_auc_score(y, -fwd), float(np.mean(fwd[: len(later)] - ks))))
    gaps_all = np.concatenate(gaps)
    out = [f"[reach_qm elo] {len(own)} held-out queries; the same forward pairs scored at every level (the Elo token set to the level):",
           f"  {'level':>7}{'mean d for k<=10':>18}{'k 11-25':>10}{'k 26-40':>10}{'reverse-of-time AUC':>22}"]
    for e in levels:
        d = np.concatenate(pred[e])
        out.append(f"  {e:>7}{d[gaps_all <= 10].mean():>18.1f}{d[(gaps_all > 10) & (gaps_all <= 25)].mean():>10.1f}{d[gaps_all > 25].mean():>10.1f}{np.mean(aucs[e]):>22.3f}")
    o = np.array(own)
    out.append("  at each query's OWN Elo, by band: " + "; ".join(f"{lo}-{hi}: AUC {o[(o[:, 0] >= lo) & (o[:, 0] < hi), 1].mean():.3f}, bias {o[(o[:, 0] >= lo) & (o[:, 0] < hi), 2].mean():+.1f} plies (n {int(((o[:, 0] >= lo) & (o[:, 0] < hi)).sum())})"
                                                   for lo, hi in ELO_BANDS if ((o[:, 0] >= lo) & (o[:, 0] < hi)).sum() >= 5))
    text = "\n".join(out); log(text); return text


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["prepare", "train", "gauge", "hard", "elo"]); ap.add_argument("--pgn", default="data/external/lichess/lichess_2015-06.pgn.zst")
    ap.add_argument("--games", type=int, default=60000); ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--max-games", type=int, default=0); ap.add_argument("--d", type=int, default=64); ap.add_argument("--layers", type=int, default=3)
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        prepare(a.pgn, a.games)
    elif a.cmd == "train":
        train(a.epochs, d=a.d, layers=a.layers, max_games=a.max_games)
    elif a.cmd == "hard":
        gauge_hard()
    elif a.cmd == "elo":
        gauge_elo()
    else:
        gauge()


if __name__ == "__main__":
    main()
