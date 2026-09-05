"""drill.py -- your own errors as a tactics deck (Kaveh 2026-09-05). Reads the review's .jsonl rows and writes one
self-contained HTML page: each position is either PUNISH (the position after your error, from the opponent's side;
the answer is the refutation the review named) or FIND (your position; the answer is the better move the review
named, kept only when Maia 1500+ finds it at least one time in ten). Spaced repetition lives in the browser
(localStorage): a right answer moves a position up a box (due again in 1, 3, 7, 21 days), a wrong one back to now.
    python -m catspace.incubator.analyst.drill --jsonl tmp/review_rapid_structure.jsonl --out tmp/drill.html"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import chess

from .horizon import winp


def build_positions(jsonl_path: str) -> list[dict]:
    rows = [json.loads(l) for l in open(jsonl_path)]
    errs = [r for r in rows if r["verdict"] in ("mistake", "blunder") and not r.get("computer_move")]
    out = []
    for r in errs:
        b = chess.Board(r["fen"]); cost = abs(winp(r["eval_cp"]) - winp(r["eval_cp"] - r["drop"]))
        tc = r["tc"]; tc_h = tc
        if "+" in tc and tc.split("+")[0].isdigit():
            tc_h = f"{int(tc.split('+')[0]) // 60}+{tc.split('+')[1]}"
        base = {"id": f"{r['date'].replace('.', '')}-{r['game']}-{r['ply']}", "move_no": r["ply"] // 2 + 1, "played": r["san"],
                "best": r["best"], "cost_wp": round(cost), "cost_pawns": round(r["drop"] / 100, 1), "headline": r["headline"],
                "cause": r["cause"]["cause"], "kind": r["cause"]["kind"], "date": r["date"].replace(".", "-"), "tc": tc_h,
                "verdict": r["verdict"], "best_line": r.get("best_line")}
        ref = r.get("refutation")
        if ref and r["cause"]["cause"] in ("walked_into", "enabled_tactic", "ignored_threat") and (r["cause"].get("mechanisms") or []):
            b2 = b.copy(); b2.push_san(r["san"])
            out.append({**base, "mode": "punish", "fen": b2.fen(), "fen_before": r["fen"], "side": "w" if b2.turn else "b",
                        "answer": ref["san"], "line": ref["line"], "p_best": {k: round(v * 100) for k, v in r["p_played"].items()}})
        pb = r["p_best"]
        if r["cause"]["cause"] == "missed_better" and max(pb.get("1500", 0), pb.get("1700", 0), pb.get("1900", 0)) >= 0.10 and " -- " in r["headline"]:
            out.append({**base, "mode": "find", "fen": r["fen"], "side": "w" if b.turn else "b", "answer": r["best"],
                        "line": r.get("best_line"), "p_best": {k: round(v * 100) for k, v in pb.items()}})
    out.sort(key=lambda x: -x["cost_wp"])
    return out


PAGE = r"""<title>Huvaxshtra Drill</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Noto+Sans+Symbols+2&display=swap">
<style>
:root {
  --ground: #EDEEEA; --panel: #F7F7F4; --ink: #1B1F1D; --muted: #5E6661; --rule: #D3D6D0;
  --sq-light: #E9E2C9; --sq-dark: #7A9A78; --sq-sel: rgba(31,95,139,.55); --sq-hint: rgba(31,95,139,.35); --sq-last: rgba(196,140,40,.45);
  --accent: #1F5F8B; --accent-ink: #FFFFFF; --good: #2E7D4F; --bad: #B5432C; --bad-soft: #F3DBD4; --good-soft: #D9EBDF;
  --piece-w: #FAFAF6; --piece-b: #1A1D1B;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #15181A; --panel: #1D2124; --ink: #E6E5DE; --muted: #9AA29C; --rule: #2E3437;
    --sq-light: #C9C1A2; --sq-dark: #587A57; --sq-sel: rgba(127,178,216,.6); --sq-hint: rgba(127,178,216,.4); --sq-last: rgba(222,170,70,.45);
    --accent: #7FB2D8; --accent-ink: #10202B; --good: #6CC08B; --bad: #E0705A; --bad-soft: #3A2521; --good-soft: #1F3328;
    --piece-w: #FAFAF6; --piece-b: #17191A;
  }
}
:root[data-theme="dark"] {
  --ground: #15181A; --panel: #1D2124; --ink: #E6E5DE; --muted: #9AA29C; --rule: #2E3437;
  --sq-light: #C9C1A2; --sq-dark: #587A57; --sq-sel: rgba(127,178,216,.6); --sq-hint: rgba(127,178,216,.4); --sq-last: rgba(222,170,70,.45);
  --accent: #7FB2D8; --accent-ink: #10202B; --good: #6CC08B; --bad: #E0705A; --bad-soft: #3A2521; --good-soft: #1F3328;
  --piece-w: #FAFAF6; --piece-b: #17191A;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--ground); color: var(--ink); font: 15px/1.5 "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif; }
.mono { font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace; font-variant-numeric: tabular-nums; }
header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 24px; padding: 20px 24px 12px; border-bottom: 1px solid var(--rule); }
header h1 { margin: 0; font: 700 26px/1.1 "Bricolage Grotesque", "IBM Plex Sans", system-ui, sans-serif; letter-spacing: -.01em; text-wrap: balance; }
header .sub { color: var(--muted); font-size: 13px; }
.stats { display: flex; gap: 18px; margin-left: auto; font-size: 13px; color: var(--muted); }
.stats b { color: var(--ink); font-weight: 600; }
main { display: grid; grid-template-columns: minmax(280px, 520px) minmax(300px, 1fr); gap: 24px; padding: 20px 24px; max-width: 1100px; }
@media (max-width: 760px) { main { grid-template-columns: 1fr; padding: 14px; } header { padding: 14px; } .stats { margin-left: 0; } }
.board { display: grid; grid-template-columns: repeat(8, 1fr); aspect-ratio: 1; width: 100%; border: 2px solid var(--sq-dark); border-radius: 3px; overflow: hidden; user-select: none; }
.sq { position: relative; display: flex; align-items: center; justify-content: center; font-family: "Noto Sans Symbols 2", "Segoe UI Symbol", "Apple Symbols", "DejaVu Sans", sans-serif; font-size: clamp(22px, 7.2vw, 46px); line-height: 1; cursor: pointer; }
.sq.l { background: var(--sq-light); } .sq.d { background: var(--sq-dark); }
.sq.sel::before, .sq.last::before, .sq.hint::after { content: ""; position: absolute; inset: 0; pointer-events: none; }
.sq.sel::before { background: var(--sq-sel); } .sq.last::before { background: var(--sq-last); }
.sq.hint::after { inset: 36%; border-radius: 50%; background: var(--sq-hint); }
.sq.hint.cap::after { inset: 4%; border-radius: 50%; background: transparent; border: 4px solid var(--sq-hint); }
.sq .p { position: relative; z-index: 1; }
.sq .p.w { color: var(--piece-w); text-shadow: 0 0 1px #000, 0 1px 1px rgba(0,0,0,.6); } .sq .p.b { color: var(--piece-b); }
.sq .c { position: absolute; left: 3px; top: 1px; font: 500 10px/1 "IBM Plex Mono", monospace; color: rgba(0,0,0,.45); }
.panel { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
.eyebrow { font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.mode { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; }
.mode.punish { background: var(--bad-soft); color: var(--bad); } .mode.find { background: var(--good-soft); color: var(--good); }
.prompt { font: 500 22px/1.25 "Bricolage Grotesque", "IBM Plex Sans", sans-serif; text-wrap: balance; margin: 0; }
.context { color: var(--muted); }
.entry { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.entry input { font: 15px "IBM Plex Mono", monospace; padding: 8px 10px; width: 9ch; border: 1px solid var(--rule); border-radius: 6px; background: var(--panel); color: var(--ink); }
button { font: 500 14px "IBM Plex Sans", system-ui, sans-serif; padding: 8px 14px; border-radius: 6px; border: 1px solid var(--rule); background: var(--panel); color: var(--ink); cursor: pointer; }
button.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
button:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.result { border-radius: 8px; padding: 14px 16px; background: var(--panel); border: 1px solid var(--rule); }
.result.ok { border-color: var(--good); } .result.no { border-color: var(--bad); }
.result h2 { margin: 0 0 6px; font: 600 16px/1.3 "IBM Plex Sans", sans-serif; }
.result p { margin: 6px 0; }
.line { color: var(--muted); }
.ladder { display: grid; grid-template-columns: 5ch 1fr 4ch; gap: 4px 8px; align-items: center; font-size: 12px; margin-top: 8px; }
.ladder .bar { height: 8px; background: var(--rule); border-radius: 4px; overflow: hidden; }
.ladder .bar i { display: block; height: 100%; background: var(--accent); }
.deck { padding: 0 24px 40px; max-width: 1100px; }
@media (max-width: 760px) { .deck { padding: 0 14px 40px; } }
.deck h2 { font: 700 18px/1.2 "Bricolage Grotesque", sans-serif; margin: 8px 0 10px; }
.deck table { border-collapse: collapse; width: 100%; font-size: 13px; }
.deck th { text-align: left; font-weight: 500; color: var(--muted); font-size: 12px; letter-spacing: .04em; text-transform: uppercase; padding: 6px 8px; border-bottom: 1px solid var(--rule); }
.deck td { padding: 6px 8px; border-bottom: 1px solid var(--rule); vertical-align: top; }
.deck tr.cur td { background: var(--panel); }
.deck tr { cursor: pointer; }
.pill { display: inline-block; padding: 1px 7px; border-radius: 999px; font-size: 11px; font-weight: 600; letter-spacing: .03em; }
.pill.due { background: var(--bad-soft); color: var(--bad); } .pill.new { background: var(--panel); color: var(--muted); border: 1px solid var(--rule); }
.pill.box { background: var(--good-soft); color: var(--good); }
.cost { color: var(--bad); font-weight: 600; }
.wrap { overflow-x: auto; }
a { color: var(--accent); }
.foot { color: var(--muted); font-size: 12px; padding: 0 24px 30px; max-width: 70ch; }
@media (prefers-reduced-motion: no-preference) { .result { animation: in .18s ease-out; } @keyframes in { from { opacity: .6; transform: translateY(3px); } } }
</style>
<header>
  <div><h1>Huvaxshtra Drill</h1><div class="sub">Your own rapid mistakes, served back until they stop happening. Positions from the review of 36 games since June.</div></div>
  <div class="stats"><span>due <b id="s-due">0</b></span><span>learned <b id="s-learned">0</b></span><span>total <b id="s-total">0</b></span></div>
</header>
<main>
  <div>
    <div class="board" id="board" aria-label="chess board"></div>
    <div class="entry" style="margin-top:10px">
      <label for="san" class="eyebrow">or type the move</label>
      <input id="san" class="mono" placeholder="Nxc3" autocomplete="off" autocapitalize="off" spellcheck="false">
      <button id="go" class="primary">Answer</button>
      <button id="show">Show answer</button>
      <button id="next">Next</button>
    </div>
  </div>
  <div class="panel">
    <div><span class="mode" id="mode">punish</span> <span class="eyebrow" id="meta"></span></div>
    <p class="prompt" id="prompt"></p>
    <div class="context" id="context"></div>
    <div id="result" hidden class="result"></div>
    <div><a id="lichess" href="#" target="_blank" rel="noopener">Open this position on Lichess</a></div>
  </div>
</main>
<section class="deck">
  <h2>The deck</h2>
  <div class="wrap"><table><thead><tr><th>#</th><th>Mode</th><th>Game</th><th>Cost</th><th>Status</th><th>What happened</th></tr></thead><tbody id="deck"></tbody></table></div>
</section>
<p class="foot">Grading accepts the exact move the review named. In "find" positions other moves may also be fine; the answer shown is the engine's first choice. Costs are in win-probability points against the best move. The ladder shows how often Maia 1100–1900 plays the move in question. Progress is stored in this browser only.</p>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
<script>
const POSITIONS = __POSITIONS__;
const GLYPH = {K:"♔",Q:"♕",R:"♖",B:"♗",N:"♘",P:"♙",k:"♚",q:"♛",r:"♜",b:"♝",n:"♞",p:"♟"};
const BOXES = [0, 1, 3, 7, 21];
const KEY = "huvaxshtra-drill-v1";
let progress = {};
try { progress = JSON.parse(localStorage.getItem(KEY) || "{}") || {}; } catch (e) { progress = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(progress)); } catch (e) {} }
const now = () => Date.now();
const day = 86400000;
function status(p) { const s = progress[p.id]; if (!s) return {label: "new", cls: "new", due: true}; if (s.due <= now()) return {label: "due", cls: "due", due: true}; return {label: "box " + s.box, cls: "box", due: false}; }
function queue() {
  const due = POSITIONS.filter(p => status(p).due);
  const rest = POSITIONS.filter(p => !status(p).due).sort((a, b) => (progress[a.id].due - progress[b.id].due));
  return due.concat(rest);
}
let cur = null, sel = null, hints = [], game = null, answered = false;
const hasChess = typeof Chess === "function";
function norm(s) { return (s || "").replace(/[+#]/g, "").replace(/^0-0-0$/, "O-O-O").replace(/^0-0$/, "O-O"); }
function parseFen(fen) { const rows = fen.split(" ")[0].split("/"); const m = {}; rows.forEach((row, ri) => { let f = 0; for (const ch of row) { if (/\d/.test(ch)) f += +ch; else { m["abcdefgh"[f] + (8 - ri)] = ch; f++; } } }); return m; }
function render() {
  const board = document.getElementById("board"); board.innerHTML = "";
  const pieces = parseFen(cur.fen); const flip = cur.side === "b";
  const last = cur.mode === "punish" ? lastMoveSquares() : [];
  for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++) {
    const rank = flip ? r + 1 : 8 - r, file = flip ? 7 - f : f; const sq = "abcdefgh"[file] + rank;
    const d = document.createElement("div"); d.className = "sq " + ((file + rank) % 2 ? "l" : "d"); d.dataset.sq = sq;
    if (sq === sel) d.classList.add("sel");
    if (last.includes(sq)) d.classList.add("last");
    const h = hints.find(x => x.to === sq); if (h) { d.classList.add("hint"); if (h.captured) d.classList.add("cap"); }
    if (f === 0) { const c = document.createElement("span"); c.className = "c"; c.textContent = rank; d.appendChild(c); }
    if (r === 7) { const c = document.createElement("span"); c.className = "c"; c.style.left = "auto"; c.style.right = "3px"; c.style.top = "auto"; c.style.bottom = "1px"; c.textContent = "abcdefgh"[file]; d.appendChild(c); }
    const p = pieces[sq]; if (p) { const s = document.createElement("span"); s.className = "p " + (p === p.toUpperCase() ? "w" : "b"); s.textContent = GLYPH[p]; d.appendChild(s); }
    d.addEventListener("click", () => click(sq));
    board.appendChild(d);
  }
}
function lastMoveSquares() {
  if (!hasChess || cur.mode !== "punish") return [];
  // the error was played from the previous position; recover its squares by trying the SAN on the position before it
  try { const g = new Chess(cur.fen_before); const m = g.move(cur.played, {sloppy: true}); return m ? [m.from, m.to] : []; } catch (e) { return []; }
}
function click(sq) {
  if (answered || !hasChess) return;
  const piece = game.get(sq);
  if (sel && hints.some(h => h.to === sq)) { const m = game.move({from: sel, to: sq, promotion: "q"}); sel = null; hints = []; if (m) { game.undo(); grade(m.san); } render(); return; }
  if (piece && piece.color === game.turn()) { sel = sq; hints = game.moves({square: sq, verbose: true}); } else { sel = null; hints = []; }
  render();
}
function load(p) {
  cur = p; sel = null; hints = []; answered = false;
  if (hasChess) { game = new Chess(p.fen); }
  document.getElementById("mode").textContent = p.mode; document.getElementById("mode").className = "mode " + p.mode;
  document.getElementById("meta").textContent = `${p.tc} · ${p.date} · move ${p.move_no} · ${p.verdict}`;
  const side = p.side === "w" ? "White" : "Black";
  document.getElementById("prompt").textContent = p.mode === "punish" ? `${side} to move. Punish ${p.played}.` : `${side} to move. Find the better move.`;
  document.getElementById("context").textContent = p.mode === "punish"
    ? `You were ${side === "White" ? "Black" : "White"} here and just played ${p.played}. Play the reply that made it cost ${p.cost_wp} win-probability points.`
    : `In the game you played ${p.played}, which cost ${p.cost_wp} win-probability points.`;
  document.getElementById("result").hidden = true; document.getElementById("result").className = "result";
  document.getElementById("san").value = "";
  document.getElementById("lichess").href = "https://lichess.org/analysis/standard/" + p.fen.replace(/ /g, "_");
  render(); renderDeck(); stats();
}
function ladder(p) {
  const keys = Object.keys(p.p_best).sort(); const label = cur.mode === "punish" ? "how often that level plays your error" : "how often that level finds it";
  return `<div class="eyebrow" style="margin-top:10px">${label}</div><div class="ladder">` + keys.map(k => `<span class="mono">${k}</span><span class="bar"><i style="width:${p.p_best[k]}%"></i></span><span class="mono">${p.p_best[k]}%</span>`).join("") + `</div>`;
}
function grade(san) {
  if (answered) return; answered = true;
  const ok = norm(san) === norm(cur.answer);
  const s = progress[cur.id] || {box: 0, seen: 0, correct: 0, due: 0};
  s.seen++; if (ok) { s.correct++; s.box = Math.min(s.box + 1, BOXES.length - 1); } else { s.box = 0; }
  s.due = now() + BOXES[s.box] * day; s.last = now(); progress[cur.id] = s; save();
  showResult(ok, san);
}
function showResult(ok, san) {
  const r = document.getElementById("result"); r.hidden = false; r.className = "result " + (ok === null ? "" : ok ? "ok" : "no");
  const head = ok === null ? `The answer: ${cur.answer}` : ok ? `${san} is right.` : `${san} is not it. The answer: ${cur.answer}`;
  r.innerHTML = `<h2>${head}</h2><p>${escapeHtml(cur.headline)}</p>` + (cur.line ? `<p class="line mono">${escapeHtml(cur.line)}</p>` : "") + ladder(cur);
  if (hasChess && cur.answer) { try { const g = new Chess(cur.fen); const m = g.move(cur.answer, {sloppy: true}); if (m) { sel = null; hints = [{to: m.to}]; } } catch (e) {} }
  render(); renderDeck(); stats();
}
function escapeHtml(s) { return String(s).replace(/[&<>]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[c])); }
function stats() {
  const due = POSITIONS.filter(p => status(p).due).length; const learned = POSITIONS.filter(p => (progress[p.id] || {}).box >= 3).length;
  document.getElementById("s-due").textContent = due; document.getElementById("s-learned").textContent = learned; document.getElementById("s-total").textContent = POSITIONS.length;
}
function renderDeck() {
  const tb = document.getElementById("deck"); tb.innerHTML = "";
  POSITIONS.forEach((p, i) => {
    const st = status(p); const tr = document.createElement("tr"); if (cur && p.id === cur.id) tr.className = "cur";
    const what = (progress[p.id] && progress[p.id].seen) ? p.headline.split(" (")[0] : (p.mode === "punish" ? `you played ${p.played}` : `you played ${p.played}, there was better`);
    tr.innerHTML = `<td class="mono">${i + 1}</td><td><span class="mode ${p.mode}">${p.mode}</span></td><td class="mono">${p.date} · ${p.move_no}</td><td class="cost mono">−${p.cost_wp}</td><td><span class="pill ${st.cls}">${st.label}</span></td><td>${escapeHtml(what)}</td>`;
    tr.addEventListener("click", () => load(p)); tb.appendChild(tr);
  });
}
function typed() {
  const box = document.getElementById("san"); const v = box.value.trim(); if (!v || answered) return;
  if (hasChess) { const m = game.move(v, {sloppy: true}); if (!m) { box.setCustomValidity("Not a legal move here"); box.reportValidity(); return; } game.undo(); box.setCustomValidity(""); grade(m.san); }
  else grade(v);
}
document.getElementById("go").addEventListener("click", typed);
document.getElementById("san").addEventListener("input", e => e.target.setCustomValidity(""));
document.getElementById("san").addEventListener("keydown", e => { if (e.key === "Enter") typed(); });
document.getElementById("show").addEventListener("click", () => { if (!answered) { answered = true; const s = progress[cur.id] || {box: 0, seen: 0, correct: 0, due: 0}; s.seen++; s.box = 0; s.due = now(); s.last = now(); progress[cur.id] = s; save(); } showResult(null, ""); });
document.getElementById("next").addEventListener("click", () => { const q = queue().filter(p => !cur || p.id !== cur.id); load(q[0] || POSITIONS[0]); });
if (!hasChess) { document.getElementById("context").insertAdjacentHTML("afterend", '<div class="eyebrow">Move library did not load: type the move in the box.</div>'); }
if (POSITIONS.length) load(queue()[0]); else document.getElementById("prompt").textContent = "No positions yet: run a review first.";
</script>
"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True); ap.add_argument("--out", default="tmp/drill.html")
    a = ap.parse_args(argv)
    pos = build_positions(a.jsonl)
    html = PAGE.replace("__POSITIONS__", json.dumps(pos))
    open(a.out, "w").write(html)
    c = Counter(p["mode"] for p in pos)
    print(f"{a.out}: {len(pos)} positions ({c['punish']} punish, {c['find']} find), {len(html)//1024} KB")


if __name__ == "__main__":
    main()
