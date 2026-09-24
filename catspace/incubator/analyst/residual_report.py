"""THE RESIDUAL REPORT (Kaveh 2026-09-24: "let's use the residuals to guide the building of things"): the frames a run
rejected, ranked by unexplained exceptions, each re-evaluated so its worst exception rows can be drawn on boards.
The missing word, if there is one, should be visible in those rows rather than guessed. Nothing here proposes a
primitive; it shows where the vocabulary ran out."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import chess
import chess.svg

from .lemma_gen import Frame, Generator, OUT, _popcount, guard_text, lifted, parse_frame


def worst_rows(gen: Generator, frame: Frame, slot: int, k: int = 3) -> tuple:
    """Re-fit the frame and return (lemma, [rows the guard left unexplained, most costly first])."""
    lem = gen.evaluate(frame, slot, [])
    if lem is None:
        return None, []
    rows, _ = gen.rows_of(frame, slot)
    n = len(rows); all_mask = (1 << n) - 1
    loses = sum(1 << i for i, (_, _, _, d) in enumerate(rows) if d < 0)
    exc = (all_mask & ~loses) if lem.claim in ("acting loses", "acting does not gain") else loses
    cache: dict = {}
    matched = 0
    for term in lem.full_guard():
        matched |= gen.term_mask(rows, term, cache, all_mask)
    left = [rows[i] for i in range(n) if (exc >> i) & 1 and not (matched >> i) & 1]
    left.sort(key=lambda r: -abs(r[3]))
    return lem, left[:k]


def board_svg(state, ids, move, size=300) -> str:
    b = state.board
    fill = {sq: "#f6d55c80" for sq in ids if sq is not None}; fill[move.from_square] = "#4fb3ff80"
    return chess.svg.board(b, arrows=[chess.svg.Arrow(move.from_square, move.to_square, color="#15781B")], fill=fill, size=size,
                           colors={"square light": "#f0d9b5", "square dark": "#b58863", "margin": "#00000000", "coord": "#7a4b1f"}, orientation=b.turn)


def report(residual_path: Path, dom, out_path: Path, top: int = 12, cap: int = 200, relative: bool = True, lift: bool = True, prefixes=("after", "then", "some")):
    R = json.load(open(residual_path))
    R.sort(key=lambda r: -r["residual"])
    gen = Generator(dom, cap=cap, log=lambda *a, **k: None, relative=relative, lift=lift, prefixes=prefixes, sequential=False)
    parts = ["<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Residual Report</title>"
             "<style>:root{--bg:#f7f5f0;--ink:#1e1b16;--mute:#6b655a;--line:#dcd6c8;--card:#fffdf8;--acc:#7a4b1f}@media(prefers-color-scheme:dark){:root{--bg:#16140f;--ink:#efe9dc;--mute:#a59d8c;--line:#3a3529;--card:#1f1c15;--acc:#d9a86a}}"
             "body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Georgia,serif}main{max-width:70rem;margin:0 auto;padding:1.2rem 16px 3rem}h1{font-size:1.5rem}h2{font-size:1.05rem;color:var(--acc);margin:1.6rem 0 .4rem}"
             ".g{color:var(--mute);font-size:.88rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:.8rem}.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:.6rem}svg{max-width:100%;height:auto}</style></head><body><main>"
             f"<h1>Where the vocabulary ran out</h1><p class='g'>{len(R)} frames were rejected in this run (the guard did not reach the bar). The {top} with the most unexplained exceptions, re-fitted at cap {cap}, each with its costliest unexplained rows. Frame pieces shaded, the mover blue, the move an arrow; the value is the frame's outcome relative to standing still.</p>"]
    for r in R[:top]:
        fr = parse_frame(r["frame"].replace(" (join)", ""))
        for slot in range(fr.size):
            lem, rows = worst_rows(gen, fr, slot)
            if lem is None or not rows:
                continue
            parts.append(f"<h2>{html.escape(fr.text())} :: slot {slot} ({html.escape(fr.types[slot])}) {lem.claim} in {100 * lem.holds / max(1, lem.support):.0f}% of {lem.support}; guard {html.escape(guard_text(lem.full_guard()) or '(none)')} covers {lem.covered} of {lem.covered + lem.residual}</h2><div class='grid'>")
            for state, ids, a, d in rows:
                parts.append(f"<div class='card'>{state.board.san(a[1])} value {d:+.1f}<div>{board_svg(state, ids, a[1])}</div></div>")
            parts.append("</div>")
    parts.append("</main></body></html>")
    out_path.write_text("\n".join(parts))
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--residual", required=True); ap.add_argument("--out", default=str(OUT / "residual_report.html")); ap.add_argument("--top", type=int, default=12); ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--source", default="lichess", choices=["sampled", "lichess", "games"]); ap.add_argument("--positions", type=int, default=5000); ap.add_argument("--depth", type=int, default=0)
    a = ap.parse_args(argv)
    from .chess_domain import domain
    fens = None
    if a.source == "lichess":
        import random
        rows = [json.loads(l)["fen"] for l in open("data/derived/chessbench_positions.jsonl")]
        random.Random(0).shuffle(rows); fens = rows[: a.positions]
    dom = domain(fens, oracle="frame", depth=a.depth)
    p = report(Path(a.residual), dom, Path(a.out), top=a.top, cap=a.cap)
    print(f"[residual_report] -> {p}")


if __name__ == "__main__":
    main()
