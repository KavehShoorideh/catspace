"""THE WINDOW BOARD (Kaveh 2026-10-15: "a Lichess-style interface ... back and forth ... multiple lines with Stockfish calculating live ... annotations
on different positions"): one self-contained HTML page per game -- a chessground board (Lichess's open-source board), the game as the main line, every
window's branches and the teacher's line as variations, our outcome classes as annotations on the nodes, back/forward/keyboard navigation, click a piece
to explore a new line, Stockfish (the JavaScript build) running live in a web worker with three lines, each addable as a variation.

  python -m catspace.incubator.analyst.discovery.window_board <game-id>      # reads data/derived/windows/<id>.json -> data/derived/windows/<id>_board.html

Libraries: chessground 9.1.1 (ES module, jsDelivr; its stylesheets vendored under assets/vendor/chessground-9.1.1 and inlined), chess.js 0.10.3 (cdnjs)
for legality and SAN, stockfish.js 10.0.2 (cdnjs) in a blob worker. Nothing else loads from outside the page.
"""
from __future__ import annotations

import json
import os
import sys

VENDOR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "assets", "vendor", "chessground-9.1.1")

PAGE = r"""<title>Window Board</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
/* layout: board and engine on the left, windows and the move tree on the right; one column at phone width */
:root{--bg:#f4efe7;--fg:#241f1a;--muted:#6f655a;--panel:#fffcf7;--rule:#dccfbf;--accent:#7a5230;--soft:#ede4d7;--same:#b8352a;--avoid:#2b7a3f;--other:#a66b00;--eng:#2f5f9e;
--body:'Source Sans 3',system-ui,sans-serif;--mono:'JetBrains Mono',ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1714;--fg:#ece5da;--muted:#a5998a;--panel:#231f1b;--rule:#3a332c;--accent:#c9a07a;--soft:#2c2620;--same:#f08a7e;--avoid:#6fcb8a;--other:#e3b85c;--eng:#8fb6ea;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#1a1714;--fg:#ece5da;--muted:#a5998a;--panel:#231f1b;--rule:#3a332c;--accent:#c9a07a;--soft:#2c2620;--same:#f08a7e;--avoid:#6fcb8a;--other:#e3b85c;--eng:#8fb6ea;color-scheme:dark}
%%CG_CSS%%
body{background:var(--bg);color:var(--fg);font-family:var(--body);font-size:15px;line-height:1.45}
.wrap{max-width:1240px;margin:0 auto;padding-inline:16px;padding-block:14px 40px;display:grid;gap:14px}
h1{font-size:1.25rem;font-weight:700;margin:0;text-wrap:balance}
.sub{font-family:var(--mono);font-size:.78rem;color:var(--muted)}
.grid{display:grid;grid-template-columns:minmax(280px,520px) minmax(0,1fr);gap:20px;align-items:start}
@media (max-width:860px){.grid{grid-template-columns:minmax(0,1fr)}}
.left{display:grid;gap:10px;min-width:0}
.right{display:grid;gap:12px;min-width:0}
.cg-wrap{width:100%;max-width:520px;aspect-ratio:1/1;max-width:100%}
.ctl{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
button{font:inherit;font-size:.9rem;background:var(--panel);color:var(--fg);border:1px solid var(--rule);border-radius:6px;padding:5px 11px;cursor:pointer}
button:hover{border-color:var(--accent)}button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
button.on{background:var(--accent);color:var(--bg);border-color:var(--accent)}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:10px 12px;display:grid;gap:6px;min-width:0}
.card h3{margin:0;font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:600}
.eval{font-family:var(--mono);font-size:1.05rem;font-weight:600;display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
.eval .d{font-size:.78rem;color:var(--muted);font-weight:400}
.lines{display:grid;gap:4px}
.line{font-family:var(--mono);font-size:.8rem;display:flex;gap:8px;align-items:baseline;cursor:pointer;padding:3px 6px;border-radius:4px;min-width:0}
.line:hover{background:var(--soft)}
.line b{color:var(--eng);min-width:4.5em;font-variant-numeric:tabular-nums}
.line span{overflow-wrap:anywhere}
.win{display:grid;gap:4px;padding:8px 10px;border-radius:6px;border:1px solid var(--rule);background:var(--bg)}
.win .t{font-weight:600}.win .m{font-size:.82rem;color:var(--muted)}
.chip{display:inline-block;font-family:var(--mono);font-size:.7rem;padding:1px 6px;border-radius:999px;border:1px solid currentColor;margin-left:4px;vertical-align:middle}
.SAME{color:var(--same)}.AVOIDED{color:var(--avoid)}.OTHER{color:var(--other)}.ENGINE{color:var(--eng)}.MANUAL{color:var(--muted)}.TEACHER{color:var(--eng)}
.tree{font-family:var(--mono);font-size:.86rem;line-height:1.7;max-height:60vh;overflow:auto;min-width:0}
.mv{display:inline-block;padding:0 4px;border-radius:4px;cursor:pointer}
.mv:hover{background:var(--soft)}.mv.cur{background:var(--accent);color:var(--bg)}
.mv.main{font-weight:600}.mv.manual{font-style:italic}
.num{color:var(--muted);margin-left:2px}
.var{display:block;padding-left:14px;border-left:2px solid var(--rule);margin:2px 0 2px 6px;font-size:.8rem;font-weight:400}
.ann{display:grid;gap:4px;font-size:.9rem}
.ann .k{color:var(--muted);font-size:.78rem}
textarea{font:inherit;font-size:.85rem;background:var(--bg);color:var(--fg);border:1px solid var(--rule);border-radius:6px;padding:6px;width:100%;min-height:52px;box-sizing:border-box}
.small{font-size:.82rem;color:var(--muted)}
.key{display:flex;gap:14px;flex-wrap:wrap;font-size:.8rem;color:var(--muted)}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
<div class="wrap">
<div><h1 id="title"></h1><div class="sub" id="sub"></div></div>
<div class="grid">
<div class="left">
  <div class="cg-wrap"><div id="board" class="cg-wrap"></div></div>
  <div class="ctl">
    <button id="first" title="first (Home)">|&lt;</button><button id="prev" title="back (←)">&lt;</button><button id="next" title="forward (→)">&gt;</button><button id="last" title="last (End)">&gt;|</button>
    <button id="flip">flip</button><button id="engtoggle" class="on">engine on</button>
    <span class="small" id="pos"></span>
  </div>
  <div class="card"><h3>Stockfish, live in the page (a checker, not the measure)</h3>
    <div class="eval"><span id="ev">…</span><span class="d" id="evd"></span></div>
    <div class="lines" id="lines"><div class="small">starting the engine…</div></div>
    <div class="small">Click a line to add it as a variation and step into it.</div>
  </div>
  <div class="card"><h3>This position</h3><div class="ann" id="ann"></div><textarea id="note" placeholder="Your note on this position (kept in this browser only)"></textarea></div>
</div>
<div class="right">
  <div class="card"><h3>Windows (important observations: a material shift at the end of a turn)</h3><div id="wins" style="display:grid;gap:6px"></div>
    <div class="key"><span><span class="chip SAME">SAME</span> the victim still loses</span><span><span class="chip AVOIDED">AVOIDED</span> no such loss</span><span><span class="chip OTHER">OTHER</span> another observable</span><span><span class="chip TEACHER">T</span> the teacher's line</span><span><span class="chip ENGINE">E</span> added from the engine</span></div></div>
  <div class="card"><h3>Moves and lines (bold: the game; indented: branches, the teacher's line, your own)</h3><div class="tree" id="tree"></div></div>
</div>
</div>
</div>
<script id="data" type="application/json">%%DATA%%</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
<script type="module">
import { Chessground } from 'https://cdn.jsdelivr.net/npm/chessground@9.1.1/dist/chessground.min.js';
const D = JSON.parse(document.getElementById('data').textContent);
const $ = s => document.querySelector(s);
const esc = s => (s ?? '').toString().replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

// ---- the tree of positions: the game is the main line, everything else a variation
let NID = 0; const nodes = {};
function mk(parent, san, uci, fen) { const n = { id: 'n' + (NID++), parent, san, uci, fen, children: [], ann: [], main: false, manual: false, tag: null }; nodes[n.id] = n; if (parent) parent.children.push(n); return n; }
const root = mk(null, null, null, START); root.main = true; root.ply = 0;
const mainAt = [root];
{ const c = new Chess(); let cur = root;
  for (const san of D.san) { const m = c.move(san); if (!m) break; const n = mk(cur, san, m.from + m.to + (m.promotion || ''), c.fen()); n.main = true; n.ply = cur.ply + 1; cur = n; mainAt.push(n); } }
function addLine(node, sans, tag) {                       // follow or create the line of SAN moves from node; returns the nodes
  const out = []; let cur = node;
  for (const san of sans) {
    let child = cur.children.find(x => x.san === san);
    if (!child) { const c = new Chess(cur.fen); const m = c.move(san); if (!m) break; child = mk(cur, san, m.from + m.to + (m.promotion || ''), c.fen()); child.ply = cur.ply + 1; child.tag = tag; }
    out.push(child); cur = child;
  }
  return out;
}
D.windows.forEach((w, wi) => {
  const start = mainAt[w.start_ply]; if (!start) return;
  if (w.teacher_san && w.teacher_san.length) { const t = addLine(start, w.teacher_san, 'TEACHER'); if (t[0]) t[0].ann.push({ window: wi, kind: 'teacher', text: `the teacher's line from the start of window ${wi + 1}` }); }
  for (const p of w.plies) {
    const at = mainAt[p.ply]; if (!at) continue;
    for (const b of p.branches) {
      const line = (b.line && b.line.length && b.line[0] === b.san) ? b.line : [b.san].concat(b.line || []);
      const ns = addLine(at, line, b.played ? null : b.cls);
      if (ns[0]) ns[0].ann.push({ window: wi, k: p.k, cls: b.cls, cp: b.cp, reasonable: b.reasonable, played: b.played, other: b.other, text: `${b.cls}${b.other ? ' (' + b.other + ')' : ''}: ${p.k} ply back in window ${wi + 1}; ${b.reasonable ? 'reasonable' : 'not reasonable'}${b.played ? ', the move played' : ''}; teacher ${(b.cp / 100).toFixed(2)}` });
    }
  }
  const o = mainAt[w.observable.ply + 1]; if (o) o.ann.push({ window: wi, kind: 'shift', text: `the observation of window ${wi + 1}: ${w.observable.side} gains ${w.observable.delta} (${w.observable.kind}); inevitability ${w.inevitability} ply back; the tree stopped on ${w.stop}` });
});

// ---- the board
let cur = root; let orient = 'white';
function turnOf(fen) { return fen.split(' ')[1] === 'w' ? 'white' : 'black'; }
function dests(fen) { const c = new Chess(fen); const m = new Map(); for (const mv of c.moves({ verbose: true })) { if (!m.has(mv.from)) m.set(mv.from, []); m.get(mv.from).push(mv.to); } return m; }
const cg = Chessground($('#board'), { fen: START, orientation: orient, movable: { free: false, color: 'white', dests: dests(START), events: { after: onBoardMove } }, highlight: { lastMove: true, check: true }, animation: { duration: 150 } });
function onBoardMove(from, to) {
  const c = new Chess(cur.fen); const m = c.move({ from, to, promotion: 'q' });
  if (!m) { setBoard(); return; }
  let child = cur.children.find(x => x.san === m.san);
  if (!child) { child = mk(cur, m.san, from + to + (m.promotion || ''), c.fen()); child.ply = cur.ply + 1; child.manual = true; child.tag = 'MANUAL'; renderTree(); }
  goTo(child);
}
function setBoard() {
  const c = new Chess(cur.fen); const turn = turnOf(cur.fen);
  cg.set({ fen: cur.fen, orientation: orient, turnColor: turn, lastMove: cur.uci ? [cur.uci.slice(0, 2), cur.uci.slice(2, 4)] : undefined, check: c.in_check() ? turn : false, movable: { color: turn, dests: dests(cur.fen) } });
}

// ---- navigation
function goTo(n) {
  cur = n; setBoard();
  document.querySelectorAll('.mv.cur').forEach(e => e.classList.remove('cur'));
  const el = document.getElementById(n.id); if (el) { el.classList.add('cur'); el.scrollIntoView({ block: 'nearest' }); }
  const mv = n.ply ? `${Math.ceil(n.ply / 2)}${n.ply % 2 ? '.' : '…'} ${n.san}` : 'start';
  $('#pos').textContent = `${mv} · ${turnOf(n.fen)} to move`;
  renderAnn(); loadNote(); analyse();
}
$('#first').onclick = () => goTo(root);
$('#prev').onclick = () => { if (cur.parent) goTo(cur.parent); };
$('#next').onclick = () => { if (cur.children[0]) goTo(cur.children[0]); };
$('#last').onclick = () => { let n = cur; while (n.children[0]) n = n.children[0]; goTo(n); };
$('#flip').onclick = () => { orient = orient === 'white' ? 'black' : 'white'; setBoard(); };
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'ArrowLeft') { $('#prev').click(); e.preventDefault(); }
  if (e.key === 'ArrowRight') { $('#next').click(); e.preventDefault(); }
  if (e.key === 'Home') { $('#first').click(); e.preventDefault(); }
  if (e.key === 'End') { $('#last').click(); e.preventDefault(); }
  if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {            // the sibling variation
    const p = cur.parent; if (!p) return; const i = p.children.indexOf(cur); const j = e.key === 'ArrowUp' ? i - 1 : i + 1;
    if (p.children[j]) goTo(p.children[j]); e.preventDefault();
  }
});

// ---- the move tree (the game in bold; a sibling's line is shown right after the move it replaces, indented)
function moveEl(n) {
  const num = n.ply % 2 ? `<span class="num">${Math.ceil(n.ply / 2)}.</span>` : (n.parent && n.parent.children[0] !== n ? `<span class="num">${Math.ceil(n.ply / 2)}…</span>` : '');
  const chips = n.ann.filter(a => a.cls).map(a => `<span class="chip ${a.cls}">${a.cls}</span>`).join('') + (n.tag === 'TEACHER' ? '<span class="chip TEACHER">T</span>' : '') + (n.tag === 'ENGINE' ? '<span class="chip ENGINE">E</span>' : '');
  return `${num}<span class="mv ${n.main ? 'main' : ''} ${n.manual ? 'manual' : ''}" id="${n.id}" data-id="${n.id}">${esc(n.san)}${chips}</span> `;
}
function renderSeq(n) {
  let html = ''; let x = n;
  while (x) {
    html += moveEl(x);
    const sibs = x.parent.children.filter(s => s !== x && x.parent.children[0] === x);     // variations only off the line's first choice
    for (const s of sibs) html += `<span class="var">${renderSeq(s)}</span>`;
    x = x.children[0];
  }
  return html;
}
function renderTree() { $('#tree').innerHTML = root.children[0] ? renderSeq(root.children[0]) : ''; const el = document.getElementById(cur.id); if (el) el.classList.add('cur'); }
$('#tree').addEventListener('click', e => { const el = e.target.closest('.mv'); if (el) goTo(nodes[el.dataset.id]); });

// ---- annotations and notes
function renderAnn() {
  const a = cur.ann; let html = '';
  if (cur.manual) html += `<div><span class="chip MANUAL">your line</span></div>`;
  if (cur.tag === 'ENGINE') html += `<div><span class="chip ENGINE">E</span> added from the engine</div>`;
  for (const x of a) html += `<div>${x.cls ? `<span class="chip ${x.cls}">${x.cls}</span> ` : ''}${esc(x.text)}</div>`;
  if (!html) html = `<div class="k">No annotation on this position. Branch annotations sit on the first move of each line.</div>`;
  $('#ann').innerHTML = html;
}
const noteKey = () => 'note:' + D.game + ':' + cur.fen;
function loadNote() { try { $('#note').value = localStorage.getItem(noteKey()) || ''; } catch (e) { $('#note').value = ''; } }
$('#note').addEventListener('input', () => { try { localStorage.setItem(noteKey(), $('#note').value); } catch (e) {} });

// ---- windows
$('#wins').innerHTML = D.windows.map((w, i) => {
  const o = w.observable; const lv = w.leaves.reduce((m, l) => (m[l.cls] = (m[l.cls] || 0) + 1, m), {});
  return `<div class="win"><div class="t">Window ${i + 1}: ${o.side} gains ${o.delta} at move ${Math.ceil((o.ply + 1) / 2)}${(o.ply + 1) % 2 ? '.' : '…'} ${esc(D.san[o.ply] || '')}</div>
  <div class="m">starts at ply ${w.start_ply} (${w.depth} back) · inevitability ${w.inevitability} ply back · ${w.reasonable} reasonable branches · leaves ${Object.entries(lv).map(([k, v]) => `<span class="${k}">${k} ${v}</span>`).join(', ')}</div>
  <div class="ctl"><button data-go="${w.start_ply}">go to the start</button><button data-go="${o.ply + 1}">go to the shift</button></div></div>`;
}).join('');
$('#wins').addEventListener('click', e => { const b = e.target.closest('button[data-go]'); if (b) goTo(mainAt[+b.dataset.go] || root); });

// ---- the engine (stockfish.js, the JavaScript build, in a blob worker)
let eng = null, engOn = true, token = 0, best = {};
function startEngine() {
  try {
    const src = "importScripts('https://cdnjs.cloudflare.com/ajax/libs/stockfish.js/10.0.2/stockfish.js');";
    eng = new Worker(URL.createObjectURL(new Blob([src], { type: 'text/javascript' })));
    eng.onmessage = ev => onEngine(typeof ev.data === 'string' ? ev.data : (ev.data && ev.data.data) || '');
    eng.onerror = () => { $('#lines').innerHTML = '<div class="small">The engine could not start in this viewer. The tree and the board still work.</div>'; eng = null; };
    eng.postMessage('uci'); eng.postMessage('setoption name MultiPV value 3'); eng.postMessage('isready');
  } catch (e) { eng = null; $('#lines').innerHTML = '<div class="small">No web worker here; the engine is off.</div>'; }
}
function analyse() {
  if (!eng || !engOn) return;
  token++; best = {};
  eng.postMessage('stop'); eng.postMessage('position fen ' + cur.fen); eng.postMessage('go depth 18');
  $('#evd').textContent = 'depth 0';
}
function toSan(fen, pv) { const c = new Chess(fen); const out = []; for (const u of pv) { const m = c.move({ from: u.slice(0, 2), to: u.slice(2, 4), promotion: u[4] }); if (!m) break; out.push(m.san); } return out; }
function onEngine(line) {
  if (!line.startsWith('info') || !line.includes(' pv ')) return;
  const d = /depth (\d+)/.exec(line), mp = /multipv (\d+)/.exec(line), sc = /score (cp|mate) (-?\d+)/.exec(line), pv = / pv (.+)$/.exec(line);
  if (!d || !sc || !pv) return;
  const k = mp ? +mp[1] : 1; const white = turnOf(cur.fen) === 'white'; const v = +sc[2] * (white ? 1 : -1);
  best[k] = { depth: +d[1], mate: sc[1] === 'mate', v, pv: pv[1].split(' ') };
  const b1 = best[1]; if (b1) { $('#ev').textContent = b1.mate ? `#${b1.v}` : `${b1.v > 0 ? '+' : ''}${(b1.v / 100).toFixed(2)}`; $('#evd').textContent = `depth ${b1.depth}, White's view`; }
  $('#lines').innerHTML = Object.keys(best).sort().map(i => { const b = best[i]; const sans = toSan(cur.fen, b.pv.slice(0, 10)); return `<div class="line" data-pv="${esc(b.pv.slice(0, 10).join(' '))}"><b>${b.mate ? '#' + b.v : (b.v > 0 ? '+' : '') + (b.v / 100).toFixed(2)}</b><span>${esc(sans.join(' '))}</span></div>`; }).join('');
}
$('#lines').addEventListener('click', e => { const el = e.target.closest('.line'); if (!el || !el.dataset.pv) return; const sans = toSan(cur.fen, el.dataset.pv.split(' ')); const ns = addLine(cur, sans, 'ENGINE'); if (ns[0]) { ns[0].ann.push({ text: `added from the engine at depth ${best[1] ? best[1].depth : '?'}` }); renderTree(); goTo(ns[0]); } });
$('#engtoggle').onclick = () => { engOn = !engOn; $('#engtoggle').textContent = engOn ? 'engine on' : 'engine off'; $('#engtoggle').classList.toggle('on', engOn); if (engOn) analyse(); else if (eng) { eng.postMessage('stop'); $('#lines').innerHTML = '<div class="small">engine off</div>'; } };

// ---- start
$('#title').textContent = `${D.white} (${D.elo[0]}) v ${D.black} (${D.elo[1]})`;
$('#sub').textContent = `Lichess ${D.game} · band ${D.band} · ${D.windows.length} windows · teacher depth ${D.knobs.depth}, multipv ${D.knobs.multipv}, margin ${JSON.stringify(D.knobs.margin)} cp, branch cap ${D.knobs.branches}`;
renderTree(); startEngine();
goTo(D.windows[0] ? (mainAt[D.windows[0].start_ply] || root) : root);
</script>
"""


def write_board(game_id: str) -> str:
    from catspace.io import paths
    data = json.load(open(paths.derived(f"windows/{game_id}.json")))
    css = "\n".join(open(os.path.join(VENDOR, f)).read() for f in ("cg_base.css", "cg_brown.css", "cg_cburnett.css"))
    html = PAGE.replace("%%CG_CSS%%", css).replace("%%DATA%%", json.dumps(data).replace("</", "<\\/"))
    out = paths.derived(f"windows/{game_id}_board.html")
    open(out, "w").write(html)
    print(f"OBS window board {game_id}: {len(data['san'])} plies, {len(data['windows'])} windows, {len(html) // 1024} KB -> {out}")
    return out


if __name__ == "__main__":
    write_board(sys.argv[1])
