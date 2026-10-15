"""The WINDOW PAGE: one static HTML per game (boards from python-chess SVG, data inline as JSON, no external libraries). Written by windows.py --page."""
from __future__ import annotations

import json
import os
from urllib.parse import quote

CSS = """
:root{--bg:#fafaf7;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#d9d9d2;--same:#b3261e;--avoid:#1b7a3a;--other:#9a6700;--acc:#2456c9}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#15161a;--fg:#e8e8e3;--mut:#9a9a94;--card:#1e2026;--line:#33353d;--same:#ff8a80;--avoid:#6fd08c;--other:#e6b450;--acc:#8ab4ff}}
:root[data-theme=dark]{--bg:#15161a;--fg:#e8e8e3;--mut:#9a9a94;--card:#1e2026;--line:#33353d;--same:#ff8a80;--avoid:#6fd08c;--other:#e6b450;--acc:#8ab4ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;padding:16px;overflow-x:hidden}
h1{font-size:1.2rem;margin:0 0 4px}h2{font-size:1rem;margin:18px 0 6px}.mut{color:var(--mut);font-size:.85rem}
.wrap{display:flex;gap:16px;flex-wrap:wrap;margin-top:12px}.left{flex:0 0 360px;max-width:100%;position:sticky;top:8px;align-self:flex-start}
.left svg{width:100%;height:auto;max-width:360px}.right{flex:1 1 300px;min-width:0}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin-bottom:10px}
.row{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:4px 0}.k{width:4.5em;color:var(--mut);font-size:.8rem}
button.b{font:inherit;border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:6px;padding:3px 8px;cursor:pointer}
button.b.sel{outline:2px solid var(--acc)}button.b.played{font-weight:700}button.b.unr{opacity:.55;border-style:dashed}
.SAME{color:var(--same)}.AVOIDED{color:var(--avoid)}.OTHER{color:var(--other)}.tag{font-size:.7rem;margin-left:4px}
.lk{margin-right:8px;font-size:.8rem;color:var(--acc);overflow-wrap:anywhere}.lks{display:inline-block;margin-right:10px}.pill.div{border-color:var(--same);font-weight:700}.pill{display:inline-block;border:1px solid var(--line);border-radius:10px;padding:0 8px;margin:0 4px 4px 0;font-size:.8rem}
@media(max-width:700px){.left{position:static;flex-basis:100%}}
"""

JS = """
const D=JSON.parse(document.getElementById('data').textContent);
const $=(s)=>document.querySelector(s);let cur=null;
function show(key,cap){$('#board').innerHTML=D.svgs[key];$('#cap').textContent=cap;document.querySelectorAll('button.b').forEach(b=>b.classList.toggle('sel',b.dataset.key===key));}
function el(t,c,x){const e=document.createElement(t);if(c)e.className=c;if(x!==undefined)e.textContent=x;return e;}
const R=$('#right');
function lk(t,u){const a=el('a','lk',t);a.href=u;a.target='_blank';a.rel='noopener';return a;}
function links(L,pre){const d=el('span','lks');if(pre)d.appendChild(el('span','mut',pre+' '));d.appendChild(lk('game at this ply',L.game));d.appendChild(lk('analyse from here',L.fen));d.appendChild(lk('analyse this line',L.line));return d;}
function lineRow(name,san,div,L){const r=el('div','row');r.appendChild(el('span','k',name));san.forEach((s,i)=>{const x=el('span','pill'+(div!==null&&i>=div?' div':''),s);r.appendChild(x);});if(div!==null)r.appendChild(el('span','mut','(diverge at move '+(div+1)+')'));r.appendChild(links(L));return r;}
D.windows.forEach((w,n)=>{const o=w.observable,c=el('div','card');
 c.appendChild(el('h2','',`Window ${n+1}: ${o.side} ${o.kind==='mate'?'mates':'gains '+Math.abs(o.delta)+' ('+o.kind+')'} at ply ${o.ply+1}`));
 c.appendChild(el('div','mut',`inevitability: ${w.inevitability} ply(s) back (strict, all sides: ${w.inevitability_strict}) | depth ${w.depth} back | reasonable branches ${w.reasonable} | stop: ${w.stop} | cashed: ${w.cashed?'yes (the loss fell in the game)':'no'}`));
c.appendChild(lineRow('played',w.played_san,w.diverge,w.links_played));c.appendChild(lineRow('teacher',w.teacher_san,w.diverge,w.links_teacher));
 c.appendChild(links(w.links_start,'start position:'));
 const s=el('button','b','start position');s.className='b';s.dataset.key=w.start_key;s.onclick=()=>show(w.start_key,'start of the window (ply '+(w.plies[w.plies.length-1].ply+1)+')');c.appendChild(s);
 w.plies.slice().reverse().forEach(p=>{const r=el('div','row');r.appendChild(el('span','k',`${p.k} back`));r.appendChild(el('span','mut',`${p.mover} (ply ${p.ply+1})`));
  p.branches.forEach(b=>{const x=el('button','b'+(b.played?' played':'')+(b.reasonable?'':' unr'));x.dataset.key=b.key;
   x.innerHTML='';x.appendChild(document.createTextNode((b.played?'\\u25B6 ':'')+b.san));const t=el('span','tag '+b.cls,b.cls+(b.other?' ('+b.other+')':'')+' '+(b.cp/100).toFixed(2));x.appendChild(t);
   x.onclick=()=>show(b.key,`${b.san}: ${b.cls}${b.other?' ('+b.other+')':''}; teacher ${b.cp} cp (checker only); line: ${b.line.join(' ')}`);r.appendChild(x);});c.appendChild(r);const lr=el('div','row');lr.appendChild(el('span','k',''));p.branches.forEach(b=>{const q=links(b.links,b.san+':');q.className='lks';lr.appendChild(q);});c.appendChild(lr);});
 R.appendChild(c);});
const L=el('div','card');L.appendChild(el('h2','','Leaves by outcome for the side to move at each start position'));
D.merged.forEach(g=>{L.appendChild(el('div','mut',`start: ${g.start_fen} (windows ${g.windows.map(i=>i+1).join(', ')})`));
 ['winning','neutral','losing'].forEach(k=>{const r=el('div','row');r.appendChild(el('span','k',k+' '+g.leaves[k].length));g.leaves[k].forEach(l=>r.appendChild(el('span','pill '+l.cls,`${l.move} (w${l.window+1}, ${l.k} back)`)));L.appendChild(r);});});
R.appendChild(L);
const f=D.windows[0];if(f)show(f.start_key,'start of the first window');
"""


LI = "https://lichess.org"


def _fenu(fen: str) -> str:
    return f"{LI}/analysis/standard/" + quote(fen.replace(" ", "_"), safe="/_-")


def _lineu(san: list) -> str:
    return f"{LI}/analysis/pgn/" + quote("_".join(san), safe="_-=+#")


def _links(game: str, ply: int, fen: str, san: list) -> dict:
    """game at this ply (0 = the start; n = after n half-moves, as Lichess counts), analyse from the position, analyse the line (the moves from move one)."""
    return dict(game=f"{LI}/{game}#{ply}", fen=_fenu(fen), line=_lineu(san))


def write_page(res: dict, domain) -> str:
    from catspace.io import paths
    svgs, data_w = {}, []
    for n, w in enumerate(res["windows"]):
        w = json.loads(json.dumps(w))
        w["start_key"] = f"s{n}"
        hist = res["san"][:w["start_ply"]]
        w["links_start"] = _links(res["game"], w["start_ply"], w["start_fen"], hist)
        w["links_played"] = _links(res["game"], w["start_ply"], w["start_fen"], hist + w["played_san"])
        w["links_teacher"] = _links(res["game"], w["start_ply"], w["start_fen"], hist + w["teacher_san"])
        svgs[f"s{n}"] = domain.board_svg(w["start_fen"])
        for p in w["plies"]:
            for j, b in enumerate(p["branches"]):
                key = f"w{n}p{p['k']}b{j}"
                b["key"] = key
                b["links"] = _links(res["game"], p["ply"], domain.apply(p["fen"], b["uci"]), res["san"][:p["ply"]] + b["line"])
                b["links"]["game"] = f"{LI}/{res['game']}#{p['ply']}"
                svgs[key] = domain.board_svg(domain.apply(p["fen"], b["uci"]), b["uci"])
                b.pop("end_fen", None)
        data_w.append(w)
    data = dict(game=res["game"], windows=data_w, merged=res["merged"], svgs=svgs)
    k = res["knobs"]
    head = (f"<h1>Windows of {res['game']}: {res['white']} ({res['elo'][0]}) v {res['black']} ({res['elo'][1]})</h1>"
            f"<div class=mut>band {res['band']} | teacher Stockfish depth {k['depth']}, multipv {k['multipv']}, margin {k['margin']} cp, "
            f"branch cap {k['branches']}, rollout {k['rollout']} plies | observables: {', '.join(f'{a} {b}' for a, b in res['counts'].items())}</div>"
            "<div class=mut>Dashed = not reasonable (played anyway). Outcome classes: SAME (victim still loses), AVOIDED, OTHER. Centipawns are the teacher's, a checker only.</div>")
    html = (f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>Windows {res['game']}</title><style>{CSS}</style></head><body>{head}"
            f"<div class=wrap><div class=left><div id=board></div><div id=cap class=mut></div></div><div class=right id=right></div></div>"
            f"<script id=data type='application/json'>{json.dumps(data).replace('</', '<\\/')}</script><script>{JS}</script></body></html>")
    out = paths.derived(f"windows/{res['game']}.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(html)
    return out
