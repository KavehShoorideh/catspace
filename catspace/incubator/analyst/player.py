"""THE MVP PLAYER (Kaveh 2026-09-23: "an MVP that plays using these rules at a respectable level: regular rules-based
analysis, searching in move space and in the graph domain, incremental").

Two searches per move, both counted:
  MOVE SPACE   a small alpha-beta over the rules' moves with a capture quiescence (the resolved exchange), material
               plus a little mobility as the leaf, iterative deepening under a node cap. No pattern, no engine.
  GRAPH SPACE  the search over bodies for every root move: the proved lemmas embedded in the frame grown one related
               piece at a time; the verdict orders the root moves and prunes the ones every lemma calls losing when
               the resolved exchange agrees. INCREMENTAL: verdicts are cached by the frame's own content (the pieces
               of the bodies, their squares, the move), so a frame met again later in the game -- or in another game
               -- costs nothing (Kaveh: "I cached the values ... in the graph space it should be incremental").
Opponent for the gauge: Maia (lc0, nodes=1, human-move sampling) at a rating; random mover as the degenerate baseline.

    python -m catspace.incubator.analyst.player play --games 10 --maia 1100 --depth 3 --nodes 4000
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import chess

from . import nodes
from .chess_domain import VAL

OUT = Path("data/derived/player")
MATE = 1000.0


def material(b: chess.Board, side: chess.Color) -> float:
    return sum(VAL[p.piece_type] for p in b.piece_map().values() if p.color == side and p.piece_type != chess.KING) - \
           sum(VAL[p.piece_type] for p in b.piece_map().values() if p.color != side and p.piece_type != chess.KING)


def mobility(b: chess.Board, side: chess.Color) -> float:
    t = b.turn
    b.turn = side; ours = b.legal_moves.count() if not b.is_check() else 0
    b.turn = not side; theirs = b.legal_moves.count() if not b.is_check() else 0
    b.turn = t
    return 0.02 * (ours - theirs)


def evaluate(b: chess.Board, side: chess.Color) -> float:
    if b.is_checkmate():
        return -MATE if b.turn == side else MATE
    if b.is_stalemate() or b.is_insufficient_material() or b.can_claim_fifty_moves():
        return 0.0
    return material(b, side) + mobility(b, side)


def mvv_lva(b: chess.Board, m: chess.Move) -> float:
    v = b.piece_at(m.to_square)
    a = b.piece_at(m.from_square)
    return (VAL[v.piece_type] * 10 - VAL[a.piece_type]) if v else (5.0 if m.promotion else 0.0)


class Search:
    """Alpha-beta with a capture/promotion quiescence, a transposition table, a node cap, iterative deepening."""
    def __init__(self, node_cap: int = 4000, q_depth: int = 6):
        self.node_cap = node_cap; self.q_depth = q_depth; self.tt: dict = {}; self.n = 0

    def quiesce(self, b: chess.Board, side: chess.Color, alpha: float, beta: float, d: int) -> float:
        self.n += 1; nodes.tick("ledger")
        stand = evaluate(b, side)
        if b.is_game_over() or d == 0 or self.n > self.node_cap:
            return stand
        to_move_us = b.turn == side
        if to_move_us:
            if stand >= beta:
                return stand
            alpha = max(alpha, stand)
        else:
            if stand <= alpha:
                return stand
            beta = min(beta, stand)
        moves = [m for m in b.legal_moves if b.is_capture(m) or m.promotion]
        if b.is_check():
            moves = list(b.legal_moves)                                       # no standing pat in check
        moves.sort(key=lambda m: -mvv_lva(b, m))
        best = stand
        for m in moves:
            b.push(m); v = self.quiesce(b, side, alpha, beta, d - 1); b.pop()
            if to_move_us:
                if v > best: best = v
                if best > alpha: alpha = best
            else:
                if v < best: best = v
                if best < beta: beta = best
            if alpha >= beta:
                break
        return best

    def alphabeta(self, b: chess.Board, side: chess.Color, depth: int, alpha: float, beta: float) -> float:
        self.n += 1; nodes.tick("ledger")
        if b.is_game_over():
            return evaluate(b, side)
        if depth == 0 or self.n > self.node_cap:
            return self.quiesce(b, side, alpha, beta, self.q_depth)
        key = (b._transposition_key(), depth, side)
        hit = self.tt.get(key)
        if hit is not None and hit[0] >= depth:
            return hit[1]
        to_move_us = b.turn == side
        moves = list(b.legal_moves)
        moves.sort(key=lambda m: -(mvv_lva(b, m) + (0.5 if b.gives_check(m) else 0.0)))
        best = -MATE * 2 if to_move_us else MATE * 2
        for m in moves:
            b.push(m); v = self.alphabeta(b, side, depth - 1, alpha, beta); b.pop()
            if to_move_us:
                if v > best: best = v
                if best > alpha: alpha = best
            else:
                if v < best: best = v
                if best < beta: beta = best
            if alpha >= beta:
                break
        self.tt[key] = (depth, best)
        return best

    def root(self, b: chess.Board, cands: list, depth: int) -> dict:
        """Iterative deepening over the candidates; returns move -> value at the deepest completed depth."""
        side = b.turn
        vals = {}
        for d in range(1, depth + 1):
            if self.n > self.node_cap:
                break
            cur = {}
            alpha, beta = -MATE * 2, MATE * 2
            for m in cands:
                b.push(m); v = self.alphabeta(b, side, d - 1, alpha, beta); b.pop()
                cur[m] = v
                alpha = max(alpha, v)
            vals = cur
        return vals


class GraphCache:
    """Incremental graph-space verdicts: keyed by the CONTENT of the frame (the bodies' pieces and squares, the
    side to move, the move), independent of the rest of the board and of the game."""
    def __init__(self, path: Path | None = None):
        self.path = path; self.table: dict = {}; self.hits = 0; self.misses = 0
        if path and path.exists():
            try:
                self.table = json.load(open(path))
            except Exception:
                self.table = {}

    @staticmethod
    def key(board: chess.Board, bodies: list, move: chess.Move) -> str:
        pcs = sorted((chess.square_name(s), board.piece_at(s).symbol()) for s in bodies)
        return json.dumps([pcs, board.turn, move.uci()])

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(self.table, open(self.path, "w"))


class Player:
    def __init__(self, library: Path | None, proved: Path | None, depth: int = 3, node_cap: int = 4000, max_bodies: int = 4, top_k: int = 8,
                 cache_path: Path | None = None, use_lemmas: bool = True, guide: bool = False, log=print):
        from .chess_domain import domain
        from .lemma_apply import Applier, load_library
        self.dom = domain()
        self.app = Applier(self.dom, load_library(library, proved)) if (use_lemmas and library) else None
        self.depth = depth; self.node_cap = node_cap; self.max_bodies = max_bodies; self.top_k = top_k; self.log = log
        self.cache = GraphCache(cache_path)
        self.ranker = None
        if guide:
            from .body_guide import GuideRanker
            self.ranker = GuideRanker(self.dom)
        self.lemma_usage = Counter(); self.stats = Counter()

    def graph_verdicts(self, b: chess.Board, moves: list) -> dict:
        """move -> (keeps, loses) over the body search, cached by frame content."""
        from .body_search import body_search
        out = {}
        for m in moves:
            r = None
            key0 = GraphCache.key(b, [m.from_square], m)
            # the cache holds the deepest level's frame key -> verdict; probe the level keys the search would grow
            r = body_search(self.app, b, m.from_square, m, max_bodies=self.max_bodies, ranker=self.ranker) if key0 not in self.cache.table else None
            if r is None:
                self.cache.hits += 1
                out[m] = tuple(self.cache.table[key0])
                continue
            self.cache.misses += 1
            last = next((l for l in reversed(r["trace"]) if l["verdict"] is not None), None)
            for l in r["trace"]:
                for idx, text, p, fired in l["used"]:
                    self.lemma_usage[text] += 1
                nodes.tick("lemma")
            verdict = (last["keeps"], last["loses"]) if last else (0, 0)
            self.cache.table[key0] = list(verdict)
            out[m] = verdict
        return out

    def choose(self, b: chess.Board) -> tuple[chess.Move, dict]:
        side = b.turn
        legal = list(b.legal_moves)
        counts = {}
        with nodes.count(counts):
            s = Search(node_cap=self.node_cap)
            # 1. the resolved exchange for every root move (cheap): the material each move guarantees after captures settle
            resolved = {}
            for m in legal:
                b.push(m); resolved[m] = s.quiesce(b, side, -MATE * 2, MATE * 2, s.q_depth); b.pop()
            # 2. graph space: lemma verdicts, cached by frame content
            gv = self.graph_verdicts(b, legal) if self.app else {}
            def prior(m):
                k, l = gv.get(m, (0, 0))
                lemma = (1.0 if k > l else -1.0 if l > k else 0.0)
                return resolved[m] + 0.5 * lemma + 0.01 * mvv_lva(b, m)
            ordered = sorted(legal, key=lambda m: -prior(m))
            # prune: moves every lemma calls losing AND the resolved exchange says lose material, unless nothing else remains
            keep = [m for m in ordered if not (gv.get(m, (0, 0))[1] > 0 and gv.get(m, (0, 0))[0] == 0 and resolved[m] < resolved[ordered[0]] - 0.5)]
            cands = (keep or ordered)[: self.top_k]
            self.stats["pruned"] += len(legal) - len(keep); self.stats["legal"] += len(legal)
            # 3. move space: alpha-beta over the candidates
            s.n = 0
            vals = s.root(b, cands, self.depth)
            best = max(cands, key=lambda m: (vals.get(m, -MATE * 3), prior(m)))
        info = dict(move=best.uci(), value=vals.get(best), resolved=resolved[best], lemma=gv.get(best), cands=[c.uci() for c in cands], nodes=nodes.total(counts), counts=counts)
        return best, info


def play_games(n: int, maia_rating: int, depth: int, node_cap: int, max_bodies: int, library: Path | None, proved: Path | None, use_lemmas: bool, guide: bool,
               plies: int = 200, seed: int = 0, opponent: str = "maia", log=print, tag: str = ""):
    from .maia import Maia
    rng = random.Random(seed)
    opp = Maia(rating=maia_rating) if opponent == "maia" else None
    sf = None
    if opponent == "stockfish":                                              # Stockfish at a UCI Elo (>= 1320), depth-limited: the ladder above Maia-1500
        import chess.engine
        sf = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")
        sf.configure({"UCI_LimitStrength": True, "UCI_Elo": max(1320, maia_rating), "Threads": 1, "Hash": 64})
    pl = Player(library, proved, depth=depth, node_cap=node_cap, max_bodies=max_bodies, cache_path=OUT / f"graph_cache{tag}.json", use_lemmas=use_lemmas, guide=guide, log=log)
    results = Counter(); node_hist = []; t_hist = []; games = []
    for g in range(n):
        b = chess.Board(); our = chess.WHITE if g % 2 == 0 else chess.BLACK
        moves = []
        while not b.is_game_over() and len(moves) < plies:
            if b.turn == our:
                t = time.perf_counter(); m, info = pl.choose(b); t_hist.append(time.perf_counter() - t); node_hist.append(info["nodes"])
            elif sf is not None:
                m = sf.play(b, chess.engine.Limit(depth=8)).move
            else:
                m = opp.sample(b, temperature=1.0, rng=rng) if opp else rng.choice(list(b.legal_moves))
            moves.append(b.san(m)); b.push(m)
        oc = b.outcome()
        if oc is None:
            mat = material(b, our)
            res = "draw(cap)" if abs(mat) < 3 else ("win(cap)" if mat > 0 else "loss(cap)")
        else:
            res = "draw" if oc.winner is None else ("win" if oc.winner == our else "loss")
        results[res] += 1
        games.append(dict(game=g, our="white" if our else "black", result=res, plies=len(moves), moves=" ".join(moves)))
        log(f"[player] game {g + 1}/{n} as {'white' if our else 'black'} vs {opponent}-{maia_rating}: {res} in {len(moves)} plies; nodes/move {sum(node_hist) / max(1, len(node_hist)):.0f}, s/move {sum(t_hist) / max(1, len(t_hist)):.2f}; graph cache hits {pl.cache.hits} misses {pl.cache.misses}")
        pl.cache.save()
    score = (results["win"] + results["win(cap)"] + 0.5 * (results["draw"] + results["draw(cap)"])) / max(1, n)
    log(f"[player] {n} games vs {opponent}-{maia_rating}, depth {depth}, node cap {node_cap}, lemmas {'on' if use_lemmas else 'off'}{' + guide' if guide else ''}: {dict(results)}; score {100 * score:.0f}%; nodes/move {sum(node_hist) / max(1, len(node_hist)):.0f}; s/move {sum(t_hist) / max(1, len(t_hist)):.2f}; pruned {pl.stats['pruned']}/{pl.stats['legal']} root moves")
    for t, k in pl.lemma_usage.most_common(6):
        log(f"    used {k:5d}  {t[:140]}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(games=games, results=dict(results), score=score, nodes_per_move=sum(node_hist) / max(1, len(node_hist)), lemma_usage=pl.lemma_usage.most_common(40)), open(OUT / f"games{tag}.json", "w"), indent=1)
    if opp:
        opp.close()
    if sf is not None:
        sf.quit()
    return score


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["play", "move"]); ap.add_argument("--games", type=int, default=10); ap.add_argument("--maia", type=int, default=1100); ap.add_argument("--opponent", choices=["maia", "random", "stockfish"], default="maia")
    ap.add_argument("--depth", type=int, default=3); ap.add_argument("--nodes", type=int, default=4000); ap.add_argument("--max-bodies", type=int, default=4); ap.add_argument("--plies", type=int, default=200)
    ap.add_argument("--library", default="data/derived/lemmas/library_games_size3.json"); ap.add_argument("--proved", default="data/derived/lemmas/proved_games_size3.json")
    ap.add_argument("--no-lemmas", action="store_true"); ap.add_argument("--guide", action="store_true"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default=""); ap.add_argument("--fen")
    a = ap.parse_args(argv)
    if a.cmd == "move":
        pl = Player(Path(a.library), Path(a.proved), depth=a.depth, node_cap=a.nodes, max_bodies=a.max_bodies, use_lemmas=not a.no_lemmas, guide=a.guide)
        b = chess.Board(a.fen or chess.STARTING_FEN); m, info = pl.choose(b); print(b.san(m), json.dumps({k: v for k, v in info.items() if k != "counts"})); return
    play_games(a.games, a.maia, a.depth, a.nodes, a.max_bodies, Path(a.library), Path(a.proved), not a.no_lemmas, a.guide, plies=a.plies, seed=a.seed, opponent=a.opponent, tag=a.tag)


if __name__ == "__main__":
    main()
