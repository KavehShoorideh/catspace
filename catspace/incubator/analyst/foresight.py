"""FORESIGHT -- imagination from experience (Kaveh 2026-09-23: "as we play games we train: these are the situations
where these lemmas fired and these gains were made; when we are close to one, our spidey senses tingle").

Labels: for every position of a played game, the GAINING lemmas (a library made with --gain) that fire for the side
to move within the next `horizon` plies of the actual continuation -- the applier run on the continuation, gains
recorded to the position where they were still in the future. Model: the graph-token transformer of the guide, the
same tokens and relation biases, with a multi-label head over the library's lemma ids: P(lemma fires within the
horizon | position). Gauge: per-lemma AUC against the lemma's base rate. Use: imagined frames searched in the order
of that probability.

    python -m catspace.incubator.analyst.foresight labels --n-games 400 --horizon 10
    python -m catspace.incubator.analyst.foresight train
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn

from .lemma_apply import Applier, load_library

OUT = Path("data/derived/lemmas")
CKPT = OUT / "foresight.pt"


def gains_fired(app: Applier, board: chess.Board, move: chess.Move) -> set:
    """Lemma ids whose gaining term fires on the move actually played (claim 'does not gain', the guard = the gain)."""
    res = app.apply(board, actor=move.from_square, only_action=move)
    out = set()
    for (e, k), vs in res["verdicts"].items():
        for c, pred, fired, ids in vs:
            if c.claim == "acting does not gain" and fired is not None:
                out.add(c.idx)
    return out


def make_labels(library: Path, proved: Path, n_games: int, horizon: int, lo: int = 8, log=print) -> Path:
    from .chess_domain import domain
    from .dataset import load as load_ds
    app = Applier(domain(), load_library(library, proved))
    meta = load_ds("data/derived/knowledge", meta_only=True)["meta"]
    by_game = defaultdict(list)
    for m in meta:
        by_game[m["game"]].append(m)
    games = sorted(by_game)[: n_games]
    t0 = time.perf_counter(); rows = []; fired_total = Counter()
    for gi, g in enumerate(games):
        plies = sorted(by_game[g], key=lambda m: m["ply"])
        # the gains fired at each ply, for the side that moved
        fired_at = {}
        for m in plies:
            b = chess.Board(m["fen"])
            try:
                mv = chess.Move.from_uci(m["played"]) if m.get("played") and len(m["played"]) in (4, 5) and m["played"][0] in "abcdefgh" else b.parse_san(m["played"])
            except Exception:
                continue
            if mv not in b.legal_moves:
                continue
            fired_at[m["ply"]] = (b.turn, gains_fired(app, b, mv))
        for m in plies:
            if m["ply"] < lo:
                continue
            side = chess.Board(m["fen"]).turn
            soon = set()
            for p in range(m["ply"], m["ply"] + horizon):
                if p in fired_at and fired_at[p][0] == side:
                    soon |= fired_at[p][1]
            rows.append(dict(fen=m["fen"], fired=sorted(soon)))
            for x in soon:
                fired_total[x] += 1
        if gi % 50 == 0 and gi:
            log(f"[foresight] {gi}/{len(games)} games, {len(rows)} positions ({time.perf_counter() - t0:.0f}s)")
    path = OUT / f"foresight_labels_h{horizon}.json"
    json.dump(dict(library=str(library), horizon=horizon, rows=rows, base_rate={k: v / max(1, len(rows)) for k, v in fired_total.items()}), open(path, "w"))
    log(f"[foresight] {len(rows)} positions from {len(games)} games; lemmas that ever fire within {horizon} plies: {len(fired_total)}; top base rates: " + ", ".join(f"{app.lemmas_by_idx[k].text[:40]}: {v / len(rows):.3f}" for k, v in fired_total.most_common(5)))
    return path


def train(labels_path: Path, epochs: int = 30, seed: int = 0, log=print):
    from .body_guide_tf import Block, MAXP, encode
    from .chess_domain import domain
    dom = domain()
    data = json.load(open(labels_path))
    rows = data["rows"]; ids = sorted({x for r in rows for x in r["fired"]})
    if not ids:
        log("[foresight] no lemma ever fired: nothing to learn (yet)"); return
    col = {x: k for k, x in enumerate(ids)}
    rng = random.Random(seed); rng.shuffle(rows)
    fens = sorted({r["fen"] for r in rows}); rng.shuffle(fens); test_f = set(fens[: len(fens) // 5])
    tr = [r for r in rows if r["fen"] not in test_f]; te = [r for r in rows if r["fen"] in test_f]
    def build(rs):
        X = []; R = []; M = []; Y = []
        for r in rs:
            b = chess.Board(r["fen"])
            mv = next(iter(b.legal_moves))                                    # no move: the mover flag is off for every token
            sqs, feat, rel = encode(dom, b, chess.Move(mv.from_square, mv.from_square), set())
            P = len(sqs); pad = MAXP - P
            X.append(np.pad(feat, ((0, pad), (0, 0)))); R.append(np.pad(rel, ((0, pad), (0, pad), (0, 0))))
            m = np.zeros(MAXP, dtype=bool); m[:P] = True; M.append(m)
            y = np.zeros(len(ids), dtype=np.float32)
            for x in r["fired"]:
                y[col[x]] = 1.0
            Y.append(y)
        return torch.tensor(np.stack(X)), torch.tensor(np.stack(R)), torch.tensor(np.stack(M)), torch.tensor(np.stack(Y))
    Xtr, Rtr, Mtr, Ytr = build(tr); Xte, Rte, Mte, Yte = build(te)
    log(f"[foresight] train {len(tr)} test {len(te)} positions; {len(ids)} lemmas that fire")

    class Foresight(nn.Module):
        def __init__(self, d_feat, n_rel, n_out, d=64, heads=4, layers=2):
            super().__init__()
            self.emb = nn.Linear(d_feat, d); self.rel_bias = nn.Linear(n_rel, heads, bias=False)
            self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)]); self.head = nn.Linear(d, n_out)

        def forward(self, feat, rel, mask):
            x = self.emb(feat); bias = self.rel_bias(rel).permute(0, 3, 1, 2)
            for blk in self.blocks:
                x = blk(x, bias, mask)
            pooled = (x * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            return self.head(pooled)
    torch.manual_seed(seed)
    model = Foresight(Xtr.shape[-1], Rtr.shape[-1], len(ids)); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pos_w = torch.tensor([(len(Ytr) - Ytr[:, k].sum()) / max(1.0, Ytr[:, k].sum()) for k in range(len(ids))]).clamp(max=50.0)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    from sklearn.metrics import roc_auc_score
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr)); tot = 0.0
        for i in range(0, len(Xtr), 64):
            idx = perm[i: i + 64]
            loss = lossf(model(Xtr[idx], Rtr[idx], Mtr[idx]), Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        if ep % 10 == 0 or ep == epochs - 1:
            with torch.no_grad():
                pe = torch.sigmoid(model(Xte, Rte, Mte)).numpy()
            aucs = []
            for k in range(len(ids)):
                y = Yte[:, k].numpy()
                if 0 < y.sum() < len(y):
                    aucs.append(roc_auc_score(y, pe[:, k]))
            log(f"[foresight] epoch {ep} train loss {tot / len(Xtr):.3f} | test: mean per-lemma AUC {np.mean(aucs):.3f} over {len(aucs)} lemmas (0.5 = the base rate alone)")
    torch.save(dict(state=model.state_dict(), ids=ids, d_feat=Xtr.shape[-1], n_rel=Rtr.shape[-1]), CKPT)
    log(f"[foresight] -> {CKPT}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["labels", "train"]); ap.add_argument("--library", default=str(OUT / "library_games_gain.json")); ap.add_argument("--proved", default=str(OUT / "proved_games_gain.json"))
    ap.add_argument("--n-games", type=int, default=400); ap.add_argument("--horizon", type=int, default=10); ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--labels")
    a = ap.parse_args(argv)
    if a.cmd == "labels":
        make_labels(Path(a.library), Path(a.proved), a.n_games, a.horizon)
    else:
        train(Path(a.labels or OUT / f"foresight_labels_h{a.horizon}.json"), a.epochs)


if __name__ == "__main__":
    main()
