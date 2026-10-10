"""ONE GAME WALKED AGAINST STOCKFISH (Kaveh 2026-10-10: start at the beginning of a game, track the useless calculations,
the state carried forward, and every disagreement with Stockfish -- then read its line). Usage: game_walk.py <game index>."""
import os, sys, json, chess, chess.engine
os.environ["RAYON_NUM_THREADS"] = "1"
import analyst_core
from catspace.incubator.analyst.waiting_frames import _games
gid, fens = list(_games("c300", 300))[int(sys.argv[1])]
eng = chess.engine.SimpleEngine.popen_uci("stockfish")
st = analyst_core.PlanState()
rows = []
for i, fen in enumerate(fens):
    b = chess.Board(fen)
    tr, dec, proofs, nodes, ms, imm, stats = st.observe(fen)
    holds = sum(1 for l in tr if "HOLDS" in l); fails = sum(1 for l in tr if ": fails" in l)
    lone = sum(1 for l in tr if "single-threat lemma" in l)
    changed = next((l for l in tr if l.startswith("what changed")), "")
    info = eng.analyse(b, chess.engine.Limit(depth=12), multipv=2)
    best = info[0]["score"].pov(b.turn).score(mate_score=3000); bm = info[0]["pv"][0]
    second = info[1]["score"].pov(b.turn).score(mate_score=3000) if len(info) > 1 else None
    mv = (dec[1] if dec[0] in ("", "defend", "cheapest") else dec[0]) if dec else None
    loss = None
    if mv:
        a = b.copy(); a.push_uci(mv)
        loss = best - eng.analyse(a, chess.engine.Limit(depth=12))["score"].pov(b.turn).score(mate_score=3000)
    only = second is not None and best - second >= 150 and best >= 150
    rows.append(dict(ply=i, fen=fen, side="W" if b.turn else "B", proofs=proofs, holds=holds, fails=fails, lone=lone, ms=ms, mem=st.size(),
                     changed=changed.count(";") + 1 if changed else 0, dec=dec and dec[3], mv=mv and b.san(chess.Move.from_uci(mv)), loss=loss,
                     best=b.san(bm), best_cp=best, only=only, line=b.variation_san(info[0]["pv"][:8]),
                     pline=(lambda a: a.variation_san(eng.analyse(a, chess.engine.Limit(depth=12))["pv"][:7]))(chess.Board(fen) if not mv else (lambda x: (x.push_uci(mv), x)[1])(chess.Board(fen))) if mv and loss and loss >= 100 else ""))
eng.quit()
json.dump(rows, open(str(__import__("catspace.io", fromlist=["paths"]).paths.experiment("game_walk.json")), "w"))
print(f"game {gid}, {len(rows)} plies")
print("ply side proofs (fail) lone  mem changes | decision -> SF best (loss)")
for r in rows:
    flag = ""
    if r["mv"] and r["loss"] is not None and r["loss"] >= 100: flag = "  <-- DISAGREE"
    elif not r["mv"] and r["only"]: flag = "  <-- MISSED (Stockfish has an only move)"
    print(f"{r['ply']:3d} {r['side']} {r['proofs']:4d} ({r['fails']:3d}) {r['lone']:3d} {r['mem']:5d} {r['changed']:3d} | {(r['mv'] or '-'):7s} -> {r['best']:7s} ({r['loss'] if r['loss'] is not None else '':>5}){flag}")
tot_p = sum(r["proofs"] for r in rows); tot_f = sum(r["fails"] for r in rows)
print(f"TOTAL proofs {tot_p}, failed {tot_f} ({100*tot_f/max(tot_p,1):.0f}%); decisions {sum(1 for r in rows if r['mv'])}; disagreements {sum(1 for r in rows if r['mv'] and r['loss'] and r['loss']>=100)}; missed only-moves {sum(1 for r in rows if not r['mv'] and r['only'])}")
