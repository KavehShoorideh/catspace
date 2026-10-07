"""PRESSURE (Kaveh 2026-10-07: "the availability of tactics is effectively pressure ... predict not the specific tactics
but the number of available tactics -- how much pressure on me, how much on them"; his example: a weakened f-pawn makes
the position sharper in the opponent's favour, and from that grows the strategic desire not to weaken it).

On the tactic census (tactic_census.py): PRESSURE on a side at a position = the number of families (of six) its opponent
has available (k = 0) against it; FUTURE pressure on us over the next 10 plies = the mean, over the opponent's five
turns, of their available families; the same for pressure on them. The questions, measured 5-fold by game:
  (1) is the pressure differential now (ours - theirs) tied to outcomes?  (the 20-ply material swing, the game result)
  (2) does STRUCTURE predict the pressure to come, beyond the pressure now and the engine's evaluation?  (ridge R2)
  (3) the f-pawn, directly: a castled king with its f-pawn home vs moved, future pressure on us.
Features from the FEN for both sides: pawn structure (tactic_census.structure), the king's shelter (holes, moved pawns,
the f-pawn, open files around the king, attackers and defenders near it, an enemy pawn storm), piece counts, mobility.

    python -m catspace.incubator.analyst.tactic_pressure --tag c300
Prints OBS lines."""

import json, sys
import numpy as np, chess
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from catspace.incubator.analyst import tactic_census_measure as M
from catspace.incubator.analyst.tactic_census import structure
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "c300"
rows = M.load(TAG); rng = np.random.default_rng(0)
by = defaultdict(dict)
for r in rows: by[r["game"]][r["ply"]] = r
FAMS = ("back_rank", "passer", "pin", "discovery", "double_threat", "material_now")
def avail(toks): return sum(1 for t in (toks or []) if t[1] == 0 and t[0] in FAMS)
def near(toks): return sum(1 for t in (toks or []) if t[1] == 1 and t[0] in FAMS)

def king_feats(b, side, tag):
    """The king's shelter and exposure for `side`."""
    k = b.king(side); x = {}
    if k is None: return x
    kf, kr = chess.square_file(k), chess.square_rank(k); fwd = 1 if side == chess.WHITE else -1
    home_rank = 1 if side == chess.WHITE else 6
    castled_k = kf >= 6 and kr == (0 if side == chess.WHITE else 7); castled_q = kf <= 2 and kr == (0 if side == chess.WHITE else 7)
    x[f"{tag}_castled_k"] = float(castled_k); x[f"{tag}_castled_q"] = float(castled_q); x[f"{tag}_king_centre"] = float(2 <= kf <= 5)
    ours = b.pieces(chess.PAWN, side); theirs = b.pieces(chess.PAWN, not side)
    holes = 0; moved = 0
    for df in (-1, 0, 1):
        f = kf + df
        if not 0 <= f < 8: continue
        ahead = [chess.square_rank(s) for s in ours if chess.square_file(s) == f and (chess.square_rank(s) - kr) * fwd > 0]
        if not ahead: holes += 1
        elif min((rr - kr) * fwd for rr in ahead) > 2: moved += 1                     # the pawn has gone far ahead
        elif all(rr != home_rank for rr in ahead): moved += 1                         # it has moved off its home rank
    x[f"{tag}_shelter_holes"] = holes; x[f"{tag}_shelter_moved"] = moved
    # the f-pawn (kingside) / c-pawn (queenside) at home, for a castled king
    fp = chess.square(5, home_rank) if castled_k else chess.square(2, home_rank) if castled_q else None
    x[f"{tag}_fpawn_home"] = float(fp is not None and fp in ours)
    x[f"{tag}_fpawn_relevant"] = float(fp is not None)
    # files around the king: open or half-open (no own pawn)
    x[f"{tag}_king_files_open"] = sum(1 for df in (-1, 0, 1) if 0 <= kf + df < 8 and not any(chess.square_file(s) == kf + df for s in ours))
    # pieces near the king
    near_sq = [s for s in chess.SQUARES if chess.square_distance(s, k) <= 2]
    x[f"{tag}_attackers_near"] = sum(1 for s in near_sq if (p := b.piece_at(s)) and p.color != side and p.piece_type != chess.PAWN)
    x[f"{tag}_defenders_near"] = sum(1 for s in near_sq if (p := b.piece_at(s)) and p.color == side and p.piece_type not in (chess.PAWN, chess.KING))
    x[f"{tag}_enemy_pawn_storm"] = sum(1 for s in theirs if abs(chess.square_file(s) - kf) <= 1 and 0 < (chess.square_rank(s) - kr) * fwd <= 3)
    return x

def feats(r, sets):
    b = chess.Board(r["fen"]); us = b.turn; x = {}
    if "now" in sets:
        x["press_on_us_now"] = avail(r["opp"]); x["press_on_them_now"] = avail(r["stm"])
        x["near_on_us_now"] = near(r["opp"]); x["near_on_them_now"] = near(r["stm"])
    if "eval" in sets:
        x["eval"] = max(-1500, min(1500, r["eval"])) / 100.0; x["material"] = r["material"]; x["phase"] = r["phase"] / 10.0
    if "structure" in sets:
        x.update({f"us_{k}": v for k, v in structure(b, us).items()}); x.update({f"them_{k}": v for k, v in structure(b, not us).items()})
        x.update(king_feats(b, us, "us")); x.update(king_feats(b, not us, "them"))
        for pt, name in ((chess.QUEEN, "q"), (chess.ROOK, "r"), (chess.BISHOP, "b"), (chess.KNIGHT, "n")):
            x[f"us_{name}"] = len(b.pieces(pt, us)); x[f"them_{name}"] = len(b.pieces(pt, not us))
        x["us_mobility"] = b.legal_moves.count()
    return x

# targets
data = []
for r in rows:
    plies = by[r["game"]]
    theirs = [plies.get(r["ply"] + d) for d in (1, 3, 5, 7, 9)]; ours = [plies.get(r["ply"] + d) for d in (2, 4, 6, 8, 10)]
    if any(p is None for p in theirs + ours) or r["opp"] is None: continue
    data.append((r, np.mean([avail(p["stm"]) for p in theirs]), np.mean([avail(p["stm"]) for p in ours])))
groups = np.array([d[0]["game"] for d in data])
y_us = np.array([d[1] for d in data]); y_them = np.array([d[2] for d in data])
print(f"OBS pressure: n {len(data)} positions; future pressure on us (their available families per turn, next 10 plies) mean {y_us.mean():.2f}, on them {y_them.mean():.2f}; "
      f"now: on us {np.mean([avail(d[0]['opp']) for d in data]):.2f}, on them {np.mean([avail(d[0]['stm']) for d in data]):.2f}")
def oof_r2(F, y):
    pred = np.zeros(len(y))
    for fit, test in GroupKFold(n_splits=5).split(F, y, groups):
        m = make_pipeline(DictVectorizer(sparse=False), StandardScaler(), Ridge(alpha=1.0)); m.fit([F[i] for i in fit], y[fit]); pred[test] = m.predict([F[i] for i in test])
    return r2_score(y, pred), pred
cache = {}
for sets in (("now",), ("eval",), ("structure",), ("now", "eval"), ("now", "structure"), ("now", "eval", "structure")):
    F = [feats(d[0], sets) for d in data]; cache[sets] = F
    r_us, _ = oof_r2(F, y_us); r_them, _ = oof_r2(F, y_them)
    print(f"OBS R2 (5-fold by game) of future pressure from {'+'.join(sets):22s}: on us {r_us:.3f}, on them {r_them:.3f}")
# the f-pawn, directly: castled kingside, our f-pawn home vs not -> future pressure on us
sel = [(d, feats(d[0], ("structure",))) for d in data]
for label, cond in (("castled kingside, f-pawn home", lambda x: x.get("us_castled_k") == 1 and x["us_fpawn_home"] == 1),
                    ("castled kingside, f-pawn moved/gone", lambda x: x.get("us_castled_k") == 1 and x["us_fpawn_home"] == 0),
                    ("castled kingside, shelter intact (0 moved, 0 holes)", lambda x: x.get("us_castled_k") == 1 and x["us_shelter_moved"] + x["us_shelter_holes"] == 0),
                    ("castled kingside, shelter broken (>= 2 moved or holes)", lambda x: x.get("us_castled_k") == 1 and x["us_shelter_moved"] + x["us_shelter_holes"] >= 2)):
    idx = [i for i, (d, x) in enumerate(sel) if cond(x)]
    if idx:
        yu = y_us[idx]; res = np.mean([sel[i][0][0]["result"] for i in idx])
        print(f"OBS   {label:52s} n {len(idx):5d}: future pressure on us {yu.mean():.2f} (now {np.mean([avail(sel[i][0][0]['opp']) for i in idx]):.2f}); game result {res:+.3f}")
# is pressure worth something? the differential now vs the 20-ply swing and the result
diff = np.array([avail(d[0]["stm"]) - avail(d[0]["opp"]) for d in data])
sw = []; res = []
for d in data:
    later = by[d[0]["game"]].get(d[0]["ply"] + 20)
    sw.append(None if later is None else later["material"] - d[0]["material"]); res.append(d[0]["result"])
sw = np.array([s if s is not None else np.nan for s in sw]); res = np.array(res)
for v in (-2, -1, 0, 1, 2):
    m = diff == v if abs(v) < 2 else (diff <= v if v < 0 else diff >= v)
    ok = m & ~np.isnan(sw)
    print(f"OBS   pressure differential now (ours - theirs) {'<=' if v == -2 else '>=' if v == 2 else '='} {v:+d}: n {m.sum():5d}; material swing >= 2 within 20 plies {100 * np.mean(sw[ok] >= 2):.0f}%, <= -2 {100 * np.mean(sw[ok] <= -2):.0f}%; result {res[m].mean():+.3f}")
