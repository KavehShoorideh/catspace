#!/usr/bin/env python
"""
experiments/viz/build_embedding_neighbors.py — "does the embedding make sense?"

Kaveh, 2026-07-13: "watch one of the games after training. At each move,
visualize the embedding by drawing more samples from the same region and
visualize those positions -- I wanna see if the embedding is even making sense."

We can't invert the encoder to synthesize arbitrary boards from an embedding,
so "draw more samples from the same region" = nearest neighbours in F-embedding
space, retrieved from a BANK of real positions (the KRRvKBP self-play shards --
same region by construction). At each White move of one played game we render:
  - the decision position (board), the move the policy chose, its reach-to-MATE_W
    and the tablebase DTZ (ground-truth win distance);
  - its k nearest neighbours by cosine similarity of F (embeddings are L2-
    normalised, so cosine = dot). Each neighbour is a rendered board annotated
    with cos, its own reach, and its tablebase DTZ.

The eyeball test: if "near in embedding" positions are genuinely SIMILAR (same
material, comparable true distance-to-mate), the embedding is meaningful here.
If the neighbours are a grab-bag with wildly different DTZ at high cosine, the
field is nonsense in this region -- which is exactly the flatness the drill-down
and reach_curvature probe quantified. A per-ply sanity line reports the Spearman
of (cosine vs -|DTZ gap|) and the fraction of neighbours sharing the exact
material, so the picture has a number under it.

Output: a self-contained HTML stepper into artifacts/generated/.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import chess
import chess.svg
import chess.syzygy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from catspace.data.encode import board_from_packed
from catspace.diagnostic_krrkbp import load_fixed_set
from catspace.nn.features import feature_planes, omega_ids


def material_sig(board):
    """Multiset of (piece_type,color) -> a hashable signature, for 'same
    material?' checks."""
    return tuple(sorted((p.piece_type, p.color) for p in board.piece_map().values()))


def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


class Bank:
    """A bank of positions embedded once, for cosine-NN retrieval + reach/DTZ."""

    @staticmethod
    def _build(fb, omega_row, z_white_t, packed, meta, device, chunk):
        import torch
        embs, reach = [], []
        with torch.no_grad():
            for i in range(0, len(packed), chunk):
                pl = torch.from_numpy(feature_planes(packed[i:i + chunk], meta[i:i + chunk])).to(device)
                om = torch.from_numpy(np.tile(omega_row, (pl.shape[0], 1))).to(device)
                f = fb.embed_F(pl, om)
                embs.append(f.cpu().numpy())
                reach.append(fb.score(f, z_white_t).cpu().numpy())
        return embs, reach

    def __init__(self, fb, omega_row, z_white_t, packed, meta, device, chunk=4096):
        self.packed, self.meta = packed, meta
        self.boards = [None] * len(packed)          # lazy-decoded on demand
        embs, reach = self._build(fb, omega_row, z_white_t, packed, meta, device, chunk)
        self.F = np.concatenate(embs, 0)            # (N,d), L2-normalised
        self.reach = np.concatenate(reach, 0)       # (N,) reach to MATE_W
        # bytes key per row for dedup / self-exclusion
        self.keys = [packed[i].tobytes() + meta[i].tobytes() for i in range(len(packed))]

    def board(self, i):
        if self.boards[i] is None:
            self.boards[i] = board_from_packed(self.packed[i], self.meta[i])
        return self.boards[i]

    def neighbours(self, f_query, k, exclude_key=None):
        sims = self.F @ f_query                      # cosine (normalised)
        order = np.argsort(-sims)
        out = []
        for i in order:
            if exclude_key is not None and self.keys[i] == exclude_key:
                continue
            out.append((int(i), float(sims[i])))
            if len(out) >= k:
                break
        return out


def load_bank_positions(shard_dirs, cap, rng):
    packed, meta = [], []
    for d in shard_dirs:
        for f in sorted(Path(d).glob("shard_*.npz")):
            z = np.load(f)
            packed.append(z["packed"]); meta.append(z["meta"])
    packed = np.concatenate(packed, 0); meta = np.concatenate(meta, 0)
    if len(packed) > cap:
        idx = rng.choice(len(packed), size=cap, replace=False)
        packed, meta = packed[idx], meta[idx]
    return packed, meta


def dtz_wdl(board, tb):
    try:
        return tb.probe_dtz(board), tb.probe_wdl(board)
    except (KeyError, chess.syzygy.MissingTableError, ValueError):
        return None, None


def dtm_line(board, tb, cap=250):
    """Plies to mate along a tablebase-optimal line: the winning side plays the
    fastest DTZ-progress move (preferring a zeroing move and avoiding repeats, so
    it can't cycle into a fivefold draw), the losing side resists (max |DTZ|).
    Returns ply count to mate, or None if drawn/unresolved. A position-property
    distance-to-mate proxy (Syzygy has DTZ/WDL, not true DTM, at 6 pieces; Kaveh
    2026-07-13: compare neighbours against DTM, not distance-to-zeroing)."""
    b = board.copy(stack=False)
    try:
        wdl0 = tb.probe_wdl(b)
    except (KeyError, chess.syzygy.MissingTableError, ValueError, IndexError):
        return None
    if wdl0 == 0:
        return None
    winner = b.turn if wdl0 > 0 else (not b.turn)
    seen = set()
    for n in range(cap):
        if b.is_checkmate():
            return n
        if b.is_stalemate() or b.is_insufficient_material():
            return None
        cand = []
        for m in b.legal_moves:
            c = b.copy(stack=False); c.push(m)
            if c.is_checkmate():
                if b.turn == winner:
                    return n + 1
                continue
            try:
                wdl = tb.probe_wdl(c); dtz = tb.probe_dtz(c)
            except (KeyError, chess.syzygy.MissingTableError, ValueError, IndexError):
                continue
            cand.append((m, c, -wdl, dtz))
        if not cand:
            return None
        if b.turn == winner:
            wins = [x for x in cand if x[2] > 0] or cand
            def wkey(x):
                m, c, mw, dtz = x
                zeroing = 0 if (b.is_capture(m) or
                                b.piece_type_at(m.from_square) == chess.PAWN) else 1
                repeat = 1 if c.board_fen() in seen else 0
                return (repeat, abs(dtz), zeroing)
            m, c, _, _ = min(wins, key=wkey)
        else:
            m, c, _, _ = max(cand, key=lambda x: abs(x[3]))
        seen.add(b.board_fen())
        b = c
    return None


def svg_board(board, lastmove=None, size=200):
    return chess.svg.board(board, lastmove=lastmove, size=size, coordinates=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="data/derived/lichess_fb_4gb_qm_plygap_only.pt")
    ap.add_argument("--bank-shards", nargs="+", default=["data/selfplay/krrkbp_loop"])
    ap.add_argument("--fixed-set", default="artifacts/experiments/krrkbp_fixed_set_n60.json")
    ap.add_argument("--index", type=int, default=0, help="which fixed KRRvKBP start")
    ap.add_argument("--k", type=int, default=6, help="neighbours per move")
    ap.add_argument("--bank-size", type=int, default=15000)
    ap.add_argument("--max-nodes", type=int, default=200)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=80)
    ap.add_argument("--opponent-skill", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--syzygy-dir", default="data/syzygy")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    import torch  # noqa: F401
    from catspace.nn.fb import load_ckpt, pick_device
    from catspace.nn.policy_fb import FBSearchPolicy
    from catspace.uci import UCIBoardPolicy

    device = pick_device(args.device)
    fb, payload = load_ckpt(Path(args.ckpt), device)
    z_white = payload["zgoals"]["MATE_W"]
    z_white_t = torch.as_tensor(z_white, dtype=torch.float32, device=device)
    pol = FBSearchPolicy(fb, z_white, max_nodes=args.max_nodes, beam=args.beam, device=device)
    omega_row = omega_ids(np.array([1800]), np.array([1800]), np.array([float("nan")]))[0]
    tb = chess.syzygy.open_tablebase(args.syzygy_dir)

    rng = np.random.default_rng(args.seed)
    from catspace.data.encode import encode_packed as board_packed, encode_meta as board_meta

    print(f"building bank from {args.bank_shards} ...", flush=True)
    bpacked, bmeta = load_bank_positions(args.bank_shards, args.bank_size, rng)
    bank = Bank(fb, omega_row, z_white_t, bpacked, bmeta, device)
    print(f"bank: {len(bank.F)} positions embedded", flush=True)

    @torch.no_grad()
    def embed_and_reach(board):
        pl = torch.from_numpy(feature_planes(np.stack([board_packed(board)]),
                                             np.stack([board_meta(board)]))).to(device)
        om = torch.from_numpy(omega_row[None]).to(device)
        f = fb.embed_F(pl, om)
        reach = float(fb.score(f, z_white_t).cpu().numpy()[0])
        return f.cpu().numpy()[0], reach

    # ---- play one game, White = policy, Black = Stockfish -----------------
    positions = load_fixed_set(args.fixed_set)
    start = positions[args.index]
    print(f"playing from index {args.index}: {start.fen()}", flush=True)
    board = start.copy(stack=False)
    rng_move = np.random.default_rng([args.seed, args.index])
    plies = []
    opp = UCIBoardPolicy(skill=args.opponent_skill, movetime=0.02)
    with opp:
        p = 0
        while p < args.max_plies and not board.is_game_over(claim_draw=True):
            # record EVERY ply (both sides), not just White's -- step per ply
            f_q, reach_q = embed_and_reach(board)
            move = pol.move(board, rng_move) if board.turn == chess.WHITE else opp.move(board, rng_move)
            san = board.san(move)
            qdtm = dtm_line(board, tb)
            _, qwdl = dtz_wdl(board, tb)
            key = board_packed(board).tobytes() + board_meta(board).tobytes()
            nbrs = []
            for i, cos in bank.neighbours(f_q, args.k, exclude_key=key):
                nb = bank.board(i)
                ndtm = dtm_line(nb, tb)
                _, nwdl = dtz_wdl(nb, tb)
                nreach = float(bank.reach[i])   # precomputed, same omega
                nbrs.append(dict(svg=svg_board(nb, size=150), cos=cos,
                                 reach=nreach, dtm=ndtm, wdl=nwdl,
                                 stm="W" if nb.turn else "B",
                                 same_mat=material_sig(nb) == material_sig(board)))
            # sanity: does embedding proximity track distance-to-mate proximity?
            gaps = [abs(n["dtm"] - qdtm) for n in nbrs if n["dtm"] is not None and qdtm is not None]
            coss = [n["cos"] for n in nbrs if n["dtm"] is not None and qdtm is not None]
            rho = _spearman(coss, [-g for g in gaps]) if len(gaps) >= 3 else float("nan")
            frac_mat = float(np.mean([n["same_mat"] for n in nbrs])) if nbrs else 0.0
            plies.append(dict(ply=p, fen=board.fen(), san=san,
                              stm="W" if board.turn == chess.WHITE else "B",
                              svg=svg_board(board, lastmove=move, size=240),
                              reach=reach_q, dtm=qdtm, wdl=qwdl,
                              nbrs=nbrs, rho=rho, frac_mat=frac_mat))
            board.push(move)
            p += 1
    outcome = board.outcome(claim_draw=True)
    result = outcome.result() if outcome else "*"
    term = outcome.termination.name if outcome else "PLY_CAP"
    tb.close()

    label = args.label or f"{Path(args.ckpt).stem}_idx{args.index}"
    out = Path(args.out) if args.out else Path("artifacts/generated") / f"embed_neighbors_{label}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(label, args, start, result, term, plies))
    print(f"\ngame: {result} ({term}) after {len(plies)} plies")
    print(f"mean neighbour-DTM-alignment rho = "
          f"{np.nanmean([p['rho'] for p in plies]):+.3f}, "
          f"mean same-material frac = {np.nanmean([p['frac_mat'] for p in plies]):.2f}")
    print(f"-> {out}")


def render_html(label, args, start, result, term, plies):
    def fmt_dtm(d, w):
        if (w or 0) == 0:
            return "dtm — (draw)" if w == 0 else "dtm —"
        if d is None:
            return "dtm —"
        tag = "win" if (w or 0) > 0 else "loss"
        return f"mate in {(d + 1) // 2} ({tag})"
    data = json.dumps([{
        "ply": p["ply"], "san": p["san"], "svg": p["svg"],
        "hdr": f"ply {p['ply']} ({p['stm']} to move) &nbsp; {html.escape(p['san'])} &nbsp; reach {p['reach']:+.3f} &nbsp; {fmt_dtm(p['dtm'], p['wdl'])}",
        "sanity": f"neighbour DTM-alignment ρ = {p['rho']:+.2f} &nbsp;·&nbsp; same-material neighbours: {p['frac_mat']*100:.0f}%",
        "nbrs": [{
            "svg": n["svg"],
            "cap": f"cos {n['cos']:.3f} · reach {n['reach']:+.2f} · {fmt_dtm(n['dtm'], n['wdl'])} · {n['stm']}"
                   + (" · ✓mat" if n["same_mat"] else " · ✗mat"),
        } for n in p["nbrs"]],
    } for p in plies])
    title = f"embedding neighbours — {html.escape(label)}"
    return f"""<!doctype html><html><head><meta charset=utf-8>
<title>{title}</title>
<style>
 body{{font:14px/1.5 -apple-system,system-ui,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}}
 header{{padding:14px 20px;border-bottom:1px solid #2a2e37;background:#161922}}
 header b{{color:#fff}} .muted{{color:#8b93a3}}
 #wrap{{display:flex;gap:28px;padding:20px;align-items:flex-start;flex-wrap:wrap}}
 #main{{flex:0 0 auto}} #main .hdr{{margin:10px 0 4px;font-size:15px}}
 .sanity{{color:#9ec7ff;font-size:13px;margin-bottom:8px}}
 #grid{{flex:1 1 480px;display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px}}
 .cell{{background:#161922;border:1px solid #2a2e37;border-radius:8px;padding:6px}}
 .cell.mat{{border-color:#2f6f43}} .cell .cap{{font-size:11px;color:#b9c0cd;margin-top:4px}}
 svg{{display:block;width:100%;height:auto;border-radius:4px}}
 #bar{{padding:12px 20px;border-top:1px solid #2a2e37;background:#161922;position:sticky;bottom:0;display:flex;gap:12px;align-items:center}}
 button{{background:#2a3140;color:#fff;border:1px solid #3a4150;border-radius:6px;padding:8px 16px;font-size:14px;cursor:pointer}}
 button:hover{{background:#374050}} #pos{{color:#8b93a3}}
 h3{{margin:0 0 8px;font-size:13px;color:#8b93a3;font-weight:600;text-transform:uppercase;letter-spacing:.04em}}
</style></head><body>
<header><b>{title}</b> &nbsp;<span class=muted>start idx {args.index} · {html.escape(start.fen())} · game {html.escape(result)} ({html.escape(term)}) · bank {args.bank_size} · k={args.k}</span><br>
<span class=muted>Left: the decision position (last move highlighted). Right: its {args.k} nearest neighbours in F-embedding space, drawn from the same-region bank. If "near in embedding" means "similar board / similar DTZ", the embedding is meaningful here.</span></header>
<div id=wrap>
 <div id=main><div class=hdr id=hdr></div><div class=sanity id=sanity></div><div id=board></div></div>
 <div style="flex:1 1 480px"><h3>nearest neighbours in embedding space</h3><div id=grid></div></div>
</div>
<div id=bar>
 <button onclick=step(-1)>◀ prev</button><button onclick=step(1)>next ▶</button>
 <span id=pos></span>
 <span class=muted style=margin-left:auto>✓mat = neighbour shares the exact material; green border = same material</span>
</div>
<script>
const D={data}; let i=0;
function draw(){{
 const p=D[i];
 document.getElementById('hdr').innerHTML=p.hdr;
 document.getElementById('sanity').innerHTML=p.sanity;
 document.getElementById('board').innerHTML=p.svg;
 document.getElementById('grid').innerHTML=p.nbrs.map(n=>
   `<div class="cell${{n.cap.includes('✓mat')?' mat':''}}">${{n.svg}}<div class=cap>${{n.cap}}</div></div>`).join('');
 document.getElementById('pos').textContent=`ply ${{p.ply}} — ${{i+1}}/${{D.length}}`;
}}
function step(d){{i=Math.max(0,Math.min(D.length-1,i+d));draw();}}
document.addEventListener('keydown',e=>{{if(e.key==='ArrowLeft')step(-1);if(e.key==='ArrowRight')step(1);}});
draw();
</script></body></html>"""


if __name__ == "__main__":
    main()
