"""PLAN 78.59 E: WHAT A ROUTE COMPARISON WOULD READ. A reading only: no move is chosen by it, nothing is wired into any player.
For ROUTES test positions drawn at random from the quiet test rows of human_heads_data (a game's own position, a move played there by the human), the move played and one other legal move chosen at random
(seeded) are made; the heads of the record are read at the two positions reached. How often the played move is the higher on each head, by band of rating: higher, equal, lower, with a 90 pct band by bootstrap over the
positions. Sign: a head's value at a reached position is for the side to move THERE (the other side); the mover's value is one less its score for the result heads (the win and the loss swap), and the negative of the
arrival head's value (what arrives for them is what does not arrive for us). The heads were fitted at quiet positions: the pairs in which BOTH reached positions are quiet are printed beside all pairs.
Cautions: what the player did is not what was forced; the move played by a stronger side is chosen with more than these heads; a head read off its quiet rows is out of its range at a position where something stands to be lost.

  .venv/bin/python -m catspace.incubator.analyst.discovery.human_heads_route --data D --record R --out-dir O [--routes N] [--threads K]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import estimator_heads as EH
from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import human_heads_result as HR

ROUTES = 200
SEED = 5959
DRAWS = 300


def legal_words(moves: list) -> list:
    import cs_engine as CE
    return list(CE.state(HD.VARIANT, None, moves)[2])


def reached(wires: list, games_meta: list, threads: int, concepts: bool):
    """Per wire the input blocks at its LAST position, for the side to move there: R (mean rating, difference, ply), K, S, C (or None), and whether the position is quiet (flags of the table) and not over."""
    from catspace.incubator.analyst.discovery import graph_layer2 as LAYER2
    from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
    model = GL.model(HD.GAME, HD.VARIANT)
    goal = G2M.goal_kind_of(HD.VARIANT)
    names, raw = model.position_table(wires, threads)
    nk = len(names)
    flat = np.frombuffer(raw, dtype=np.int16).astype(np.int32)
    w = 2 + 2 * nk
    use = [k for k, nm in enumerate(names) if nm != goal]
    _sn, stats = HD.statements_chunked([{"moves": mv} for _s, mv in wires], threads)
    off = 0
    R, K, S, Q, ok = [], [], [], [], []
    for (_s, mv), st, (welo, belo) in zip(wires, stats, games_meta):
        n = len(mv) + 1
        t = flat[off:off + n * w].reshape(n, w)
        off += n * w
        last = t[n - 1]
        first = (n - 1) % 2 == 0
        rm, ro = (welo, belo) if first else (belo, welo)
        R.append([(rm + ro) / 2, rm - ro, n - 1])
        K.append((last[2:2 + nk] - last[2 + nk:2 + 2 * nk])[use])
        row = st[n - 1] if len(st) >= n else np.full(52, np.nan)
        S.append(row[:26] - row[26:52] if not np.isnan(row).any() else np.full(26, np.nan))
        Q.append(bool(last[0] == 1 and last[1] == 1))
        ok.append(not np.isnan(row).any())
    C = None
    if concepts:
        from catspace.incubator.analyst.discovery import frames_measure as OLD
        from pathlib import Path as P
        from catspace.io import paths
        picked = OLD.pick_concepts(P(paths.experiment("unit_fates_78_54", "games_s5016_n500")), HD.CONCEPTS)
        raw_c = model.pattern_count(wires, [[len(mv)] for _s, mv in wires], [k for _dc, k, _d, _z in picked], 3, list(LAYER2.LEVELS), goal, threads)
        us, them = LAYER2.counts_matrix(raw_c, len(wires), len(picked))
        C = (us - them).toarray()
    return {"R": np.array(R, float), "K": np.array(K, float), "S": np.array(S, float), "C": C, "quiet": np.array(Q), "ok": np.array(ok)}


def assemble(blocks: dict, letters: str) -> np.ndarray:
    return np.hstack([blocks[ch] for ch in letters])


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--routes", type=int, default=ROUTES)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines: list = []

    def say(msg: str) -> None:
        lines.append(f"OBS human route: {msg}")
        print(lines[-1], flush=True)

    games = [json.loads(x) for x in (Path(a.data) / "games.jsonl").read_text().splitlines() if x.strip()]
    rows = HD.load_rows(Path(a.data) / "rows.npz")
    rec = json.loads(Path(a.record).read_text())
    rng = np.random.default_rng(SEED)
    cand = np.flatnonzero(~rows["fit"] & (rows["ply"] < np.array([len(games[g]["moves"]) for g in rows["game"]])))
    order = rng.permutation(cand)
    wires_p, wires_o, meta, band, pick, forced = [], [], [], [], [], 0
    for r in order:
        if len(pick) >= a.routes:
            break
        g = games[int(rows["game"][r])]
        p = int(rows["ply"][r])
        mv = g["moves"]
        legal = [m for m in legal_words(mv[:p]) if m != mv[p]]
        if not legal:
            forced += 1                                                  # the move played was the only legal move: no other to compare
            continue
        pick.append(r)
        other = legal[int(rng.integers(0, len(legal)))]
        wires_p.append((None, mv[:p] + [mv[p]]))
        wires_o.append((None, mv[:p] + [other]))
        meta.append((g["welo"], g["belo"]))
        band.append(g["band"])
    band = np.array(band)
    pick = np.array(pick)
    say(f"positions skipped because the move played was the only legal move: {forced}")
    say(f"routes: {len(pick)} test positions (quiet, from ply {HD.OPEN_PLIES} on) of {len(np.unique(rows['game'][pick]))} games; the move played and one other legal move chosen at random")
    bp = reached(wires_p, meta, a.threads, True)
    bo = reached(wires_o, meta, a.threads, True)
    both_ok = bp["ok"] & bo["ok"]
    both_quiet = both_ok & bp["quiet"] & bo["quiet"]
    say(f"positions reached: both readable {int(both_ok.sum())}, both quiet {int(both_quiet.sum())} (of {len(pick)})")
    heads = rec["heads"]
    spec = [("result, self-play, kinds (78.58)", "result", "K", "result"), ("result, human, kinds", "result_human_K", "K", "result"), ("result, human, ratings + kinds", "result_human_RK", "RK", "result"),
            ("result, human, ratings + kinds + statements", "result_human_RKS", "RKS", "result"), ("result, human, ratings + kinds + statements + concepts (thinned)", "result_human_RKSC_thinned", "RKSC", "result"),
            ("arrival, human, kinds", "arrival_human_K", "K", "arrival"), ("arrival, human, statements", "arrival_human_S", "S", "arrival"), ("arrival, human, kinds + statements + ratings", "arrival_human_KS_R", "KSR", "arrival"),
            ("arrival, human, kinds + statements + concepts + ratings (thinned)", "arrival_human_thin_KSC_R", "KSCR", "arrival")]
    spec = [s for s in spec if s[1] in heads]

    def value(head_name: str, letters: str, kind: str, blocks: dict) -> np.ndarray:
        x = assemble(blocks, letters)
        h = heads[head_name]
        if kind == "result":
            pr = HR.head_probs(h, x) if h.get("transform") else EH.head_predict(rec, head_name, x)
            return 1.0 - (pr[:, 0] + 0.5 * pr[:, 1])                      # the other side's score at the position reached, from the mover's side
        return -EH.head_predict(rec, head_name, x)[:, 0]

    brng = np.random.default_rng(SEED + 1)
    for label, name, letters, kind in spec:
        vp = np.where(both_ok, value(name, letters, kind, {**bp, "R": bp["R"] * 1.0}), np.nan)
        vo = np.where(both_ok, value(name, letters, kind, bo), np.nan)
        for what, mask in (("all readable pairs", both_ok), ("pairs with both quiet", both_quiet)):
            def tally(sel):
                d = vp[sel] - vo[sel]
                return int((d > 1e-9).sum()), int((np.abs(d) <= 1e-9).sum()), int((d < -1e-9).sum())
            hi, eq, lo = tally(mask)
            n = hi + eq + lo
            share = [(hi + 0.5 * eq) / n if n else float("nan")]
            idx = np.flatnonzero(mask)
            bs = []
            for _ in range(DRAWS):
                s = idx[brng.integers(0, len(idx), len(idx))] if len(idx) else idx
                h_, e_, l_ = tally(s)
                bs.append((h_ + 0.5 * e_) / max(1, h_ + e_ + l_))
            lo_b, hi_b = (np.percentile(bs, 5), np.percentile(bs, 95)) if bs else (float("nan"), float("nan"))
            say(f"[{what}] {label}: the move played is higher in {hi}, equal in {eq}, lower in {lo} of {n}; share higher (ties half) {share[0]:.3f} [{lo_b:.3f}, {hi_b:.3f}] (0.5 is chance)")
            if mask is both_ok:
                per = []
                for b, bn in enumerate(HD.BAND_NAMES):
                    h_, e_, l_ = tally(mask & (band == b))
                    per.append(f"{bn} {(h_ + 0.5 * e_) / max(1, h_ + e_ + l_):.3f} of {h_ + e_ + l_}")
                say(f"[by band of rating, all readable pairs] {label}: " + "; ".join(per))
    say(f"done in {time.perf_counter() - t0:.1f} s wall")
    (out / "route_obs.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
