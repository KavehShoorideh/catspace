"""THE COMPLETENESS / SURPRISE GAUGE (Kaveh 2026-09-23: "my best logic says this is the right move, but Stockfish
says something completely different -- that surprise is instructive; that's where we improve our lemmas").

On positions from games (the knowledge corpus: every ply of Kaveh's games with Stockfish's best move already
labelled), apply every proved lemma and measure, all beside degenerate baselines:
  SOUNDNESS   each (lemma, move) verdict against the 1-ply material outcome on the FULL board (the frames were
              sampled with only their own pieces present: this is the extrapolation, measured);
  COVERAGE    share of positions where some lemma speaks about the best move; share of legal moves spoken about;
  SURPRISE    share of positions where a proved lemma calls the best move LOSING (a contradiction: lemma bug or
              vocabulary gap), against the same rate for a random legal move;
  OUR PICK    moves with >= 1 keeps verdict and 0 loses = "what our logic allows": share of positions where the best
              move is in that set, against the set's share of the legal moves (the random baseline).
Every surprise is written out (fen, best move, the contradicting lemmas, the outcome) as the residual to mine.

    python -m catspace.incubator.analyst.gauge_lemmas --n 300 --lo 10 --hi 60
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import chess

from .lemma_apply import Applier, load_library

OUT = Path("data/derived/lemmas")


def run(n: int, lo: int, hi: int, library: Path, proved: Path, seed: int = 0, log=print, relative: bool = False):
    from .chess_domain import domain
    from .dataset import load as load_ds
    dom = domain(); app = Applier(dom, load_library(library, proved))
    meta = load_ds("data/derived/knowledge", meta_only=True)["meta"]
    rows = [m for m in meta if lo <= m["ply"] <= hi and m.get("best")]
    rng = random.Random(seed); rng.shuffle(rows)
    t0 = time.perf_counter()
    sound = Counter(); sound_by_size = defaultdict(Counter); confusion = Counter(); per_lemma = defaultdict(Counter); actual_c = Counter()
    n_pos = 0; spoken_best = 0; spoken_moves = 0; legal_moves = 0; n_emb = 0
    surprise = 0; surprise_rand = 0; pick_has_best = 0; pick_nonempty = 0; pick_share = 0.0
    contradicting = Counter(); surprises = []
    for m_ in rows:
        if n_pos >= n:
            break
        b = chess.Board(m_["fen"])
        if b.is_game_over():
            continue
        best = m_["best"]
        try:
            best_mv = chess.Move.from_uci(best) if len(best) in (4, 5) and best[0] in "abcdefgh" else b.parse_san(best)
        except Exception:
            continue
        if best_mv not in b.legal_moves:
            continue
        res = app.apply(b)
        n_pos += 1; n_emb += len(res["embeddings"])
        legal = list(b.legal_moves); legal_moves += len(legal)
        # 1-ply material outcome per move on the full board, from the mover's side
        # the outcome on a full board: material after the opponent's best capture/check reply. ABSOLUTE = minus the
        # material before the move (the generator's own label; on a full board the opponent can nearly always take
        # something, so almost every move "loses"); RELATIVE = minus what the opponent's best reply takes if we PASS
        # (the null move), i.e. what THIS move cost or won beyond the standing threats
        null_base = None
        if relative and not b.is_check():
            nb = b.copy(stack=False); nb.turn = not nb.turn
            if nb.is_valid():
                st0 = dom.state_of(b, [b.king(b.turn)])
                null_base = dom.reply_scalar(dom.state_of(nb, [b.king(b.turn)]), b.king(b.turn)) - dom.scalar(st0, b.king(b.turn))
        def outcome(mv):
            e = mv.from_square
            st = dom.state_of(b, [e])
            after = dom.transition(st, (st.board, mv))
            d = dom.reply_scalar(after, after.ids[0]) - dom.scalar(st, e)
            return d - null_base if null_base is not None else d
        verdict_of = {}
        for (e, key), vs in res["verdicts"].items():
            mv = chess.Move.from_uci(key)
            d = outcome(mv); actual = "loses" if d < 0 else "keeps"
            actual_c[actual] += 1
            for c, pred, fired, ids in vs:
                ok = pred == actual
                sound[ok] += 1; sound_by_size[c.frame.size][ok] += 1; confusion[(pred, actual)] += 1; per_lemma[c.text][ok] += 1
            verdict_of[mv] = vs
        spoken_moves += len(verdict_of)
        if best_mv in verdict_of:
            spoken_best += 1
            if any(p == "loses" for _, p, _, _ in verdict_of[best_mv]):
                surprise += 1
                d = outcome(best_mv)
                ls = [(c.text, fired, " ".join(chess.square_name(i) for i in ids)) for c, p, fired, ids in verdict_of[best_mv] if p == "loses"]
                for t, _, _ in ls:
                    contradicting[t] += 1
                surprises.append(dict(fen=b.fen(), best=best_mv.uci(), best_cp=m_.get("best_cp"), material_after_reply=d, game=m_.get("game"), ply=m_.get("ply"), lemmas=ls[:8]))
        rmv = rng.choice(legal)
        if rmv in verdict_of and any(p == "loses" for _, p, _, _ in verdict_of[rmv]):
            surprise_rand += 1
        pick = [mv for mv, vs in verdict_of.items() if all(p == "keeps" for _, p, _, _ in vs)]
        if pick:
            pick_nonempty += 1; pick_share += len(pick) / len(legal)
            if best_mv in pick:
                pick_has_best += 1
        if n_pos % 50 == 0:
            log(f"[gauge_lemmas] {n_pos}/{n} positions ({time.perf_counter() - t0:.0f}s)")
    tot = sum(sound.values())
    log(f"[gauge_lemmas] outcome = {'RELATIVE to the null move' if relative else 'ABSOLUTE (the generator label)'}")
    log(f"[gauge_lemmas] positions {n_pos}, plies {lo}-{hi}, embeddings/position {n_emb / max(1, n_pos):.1f}, {time.perf_counter() - t0:.0f}s")
    log(f"  SOUNDNESS on the full board: {sound[True]}/{tot} verdicts agree with the 1-ply outcome = {100 * sound[True] / max(1, tot):.1f}%"
        + "".join(f"; size {s}: {100 * c[True] / max(1, sum(c.values())):.1f}% of {sum(c.values())}" for s, c in sorted(sound_by_size.items())))
    log(f"  confusion (predicted -> actual): " + ", ".join(f"{p}->{a}: {k}" for (p, a), k in sorted(confusion.items())) + f"; spoken moves actually keep {actual_c['keeps']} / lose {actual_c['loses']} (majority baseline {100 * max(actual_c.values()) / max(1, sum(actual_c.values())):.1f}%)")
    worst = sorted(((c[True] / max(1, sum(c.values())), sum(c.values()), t) for t, c in per_lemma.items() if sum(c.values()) >= 30))
    log("  least sound lemmas on real boards (n >= 30):")
    for acc, k, t in worst[:8]:
        log(f"    {100 * acc:5.1f}% of {k:5d}  {t[:150]}")
    log(f"  lemmas with n >= 30: {len(worst)}; of them sound >= 80%: {sum(1 for a, _, _ in worst if a >= 0.8)}, < 50%: {sum(1 for a, _, _ in worst if a < 0.5)}")
    log(f"  COVERAGE: best move spoken about in {spoken_best}/{n_pos} = {100 * spoken_best / max(1, n_pos):.1f}% of positions; {spoken_moves}/{legal_moves} = {100 * spoken_moves / max(1, legal_moves):.1f}% of legal moves")
    log(f"  SURPRISE: best move called LOSING by >=1 proved lemma in {surprise}/{n_pos} = {100 * surprise / max(1, n_pos):.1f}% of positions; a random legal move: {surprise_rand}/{n_pos} = {100 * surprise_rand / max(1, n_pos):.1f}%")
    log(f"  OUR PICK (>=1 keeps, 0 loses): non-empty in {pick_nonempty}/{n_pos}; contains the best move in {pick_has_best}/{max(1, pick_nonempty)} = {100 * pick_has_best / max(1, pick_nonempty):.1f}%; the set is {100 * pick_share / max(1, pick_nonempty):.1f}% of the legal moves (random baseline)")
    log("  most contradicting lemmas:")
    for t, k in contradicting.most_common(8):
        log(f"    {k:4d}  {t[:170]}")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / ("surprises_relative.jsonl" if relative else "surprises.jsonl"), "w") as f:
        for s in surprises:
            f.write(json.dumps(s) + "\n")
    json.dump(dict(positions=n_pos, lo=lo, hi=hi, soundness=sound[True] / max(1, tot), soundness_n=tot, coverage_best=spoken_best / max(1, n_pos), coverage_moves=spoken_moves / max(1, legal_moves),
                   surprise=surprise / max(1, n_pos), surprise_random=surprise_rand / max(1, n_pos), pick_nonempty=pick_nonempty, pick_has_best=pick_has_best, pick_share=pick_share / max(1, pick_nonempty),
                   contradicting=contradicting.most_common(40), confusion={f"{p}->{a}": k for (p, a), k in confusion.items()},
                   per_lemma=sorted(((c[True] / max(1, sum(c.values())), sum(c.values()), t) for t, c in per_lemma.items()), key=lambda x: (x[0], -x[1]))), open(OUT / ("gauge_lemmas_relative.json" if relative else "gauge_lemmas.json"), "w"), indent=1)
    log(f"  surprises -> {OUT / 'surprises.jsonl'} ({len(surprises)})")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--lo", type=int, default=10); ap.add_argument("--hi", type=int, default=80); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--relative", action="store_true")
    ap.add_argument("--library", default=str(OUT / "library_v2_size3.json")); ap.add_argument("--proved", default=str(OUT / "proved_v2_size3.json"))
    a = ap.parse_args(argv)
    run(a.n, a.lo, a.hi, Path(a.library), Path(a.proved), seed=a.seed, relative=a.relative)


if __name__ == "__main__":
    main()
