"""TrajectoryStore -- EVERY-PLY tokenized trajectories for both populations, plus the pair sampler.

Kaveh 2026-08-05: "make sure we have entire game trajectory data for human vs human games, and also
for our synthetic sf-vs-sf games. that means every ply."

WHY THIS IS CHEAP, AND WHY IT WAS NOT BEFORE. The lc0 112-plane input costs 7168 B per position, so
every-ply over this corpus would be 8.7 TB and was never attempted -- the existing field datasets
sample FIVE plies per game. A tokenized board is 64 + 6 = 70 B, a 100x reduction, which is what makes
full trajectories tractable at all. Both populations already have complete move lists on disk, so
nothing is regenerated and Stockfish is never re-run:

  human    data/records/lichess_2019-01/*.parquet   column `moves`
  sf       data/derived/opening_pool_sfsf_moves.tsv  game_id \\t result \\t move_list
           (its move list is the FULL game from the initial position -- opening prefix plus SF
            continuation -- so replay is identical for both sources)

WHAT EVERY PLY BUYS, beyond volume. Two things the 5-ply sample could not express:
  1. ~2,300 ordered pairs per game instead of 10, so the pair set is effectively unbounded and can
     be resampled fresh each epoch rather than frozen into a dataset.
  2. REPETITIONS become visible. If a->b and later b->a both occur in real play, that pair is
     genuinely reversible -- observed evidence, not an assumption. This is the only data-grounded
     source of "these two positions are mutually reachable", and it is what lets the asymmetry in a
     quasimetric be measured against something instead of being installed by fiat.

STORAGE (Kaveh's choice): decode on the fly, keep a cached working set. The tokenizer is the
EXISTING jepa_tokenizer tokenize(); nothing new encodes a board. Prefill runs in a process pool and
the result is a flat concatenation, not a dict of arrays -- one contiguous (N,64) block indexes far
faster than 200k small arrays and is what the pair sampler wants anyway. An optional on-disk cache
keyed by (source, n_games, seed) makes a restart free; it is a CACHE, not a dataset, and deleting it
costs only the ~minutes of replay it saves.

THE FOUR PAIR KINDS (PairSampler), and exactly what each one is evidence of:

  forward     a at ply p, b at ply q > p in the SAME game. b was reached from a. Observed.
  reversible  a repetition (r1, r2) -- the same position occurring twice in one game -- makes
              (x_k -> x_r1) reachable for every k in (r1, r2]: play forward from x_k to x_r2, which
              IS x_r1. Observed, and the only backward evidence the data contains.
  backward    (x_j -> x_i) for i < j NOT covered by any repetition spanning [i, j]. Never observed
              in that direction. This is the ratchet's raw material.
  cross       two rows from DIFFERENT games. Never observed in either direction.

`backward` and `cross` are UNOBSERVED, not proven-unreachable -- a transposition can make a cross
pair genuinely reachable, and a backward pair can be reachable by a route the game did not take. So
they carry a repulsion, not a label, and the strata verdict is never "reverses are far" (which is
trained in, uniformly) but the DIFFERENTIAL between capture-crossing and quiet-reversible reverses.
"""
from __future__ import annotations

import glob
import hashlib
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from catspace.io import paths

HUMAN, SF = 0, 1
TOK_N, GLOB_N = 64, 6

# WHICH DYNAMICS IS THE BASE (Kaveh 2026-08-05): "I want to LEARN the best play, and I want to
# LEARN the human likelihood of mistake directly as a residual on top of that perfect play."
#
# So the SHARED, unconditioned field is SF -- full-strength deterministic engine play, the closest
# thing this corpus has to perfect play -- and HUMAN is a zero-initialised residual on top of it.
# This is a real semantic commitment, not a labelling detail: whichever source is id 0 becomes the
# field that every other source is expressed as a deviation FROM, so putting the human there would
# have made engine play "the deviation" and left human error baked into the base.
#
# The residual is the deliverable, not a nuisance term: it IS the human likelihood of mistake,
# per endgame and per position, measured against best play. Same construction as the M2b style
# encoder and IQEHead.pole_delta, and for the same measured reason -- a shared base makes
# representation noise COMMON to both readouts so it cancels in their difference, whereas two
# separately-fit fields disagree as much from training noise as from real dynamics.
SF_BASE, HUMAN_RESIDUAL = 0, 1
POLE_SRC = {SF: SF_BASE, HUMAN: HUMAN_RESIDUAL}
N_POLE_SOURCES = 2

# STRENGTH SCALE for the conditioning vector. Human rows carry the SIDE-TO-MOVE's real Elo; SF
# rows carry SF_ELO. That constant is a modelling choice, stated rather than buried: full-strength
# deterministic Stockfish is far above the human range, and 3500 places it ~4 sigma above the
# lichess mean on the scale below -- high enough that "best play" sits outside the human
# distribution, low enough not to blow up the normalised feature. The residual is what has to
# explain the gap, so this constant sets where the floor is ANCHORED, not how big the gap is.
SF_ELO = 3500
ELO_MEAN, ELO_SCALE = 1500.0, 500.0        # standardise: (elo - 1500) / 500


def normalise_elo(elo):
    """Elo -> the strength coordinate of the conditioning vector."""
    return (np.asarray(elo, np.float32) - ELO_MEAN) / ELO_SCALE

# ---------------------------------------------------------------------------------------------
# TERMINAL TAXONOMY (Kaveh 2026-08-05: subsumption poles "for all the win and mate types, except
# for time (flagging)").
#
# The six canonical board-caused endings are ENDINGS in clock_field.py; RESIGN is added because
# resignation IS a board-caused conversion (recorded 2026-08-05: human decisive games are 26.3%
# mate / 50.7% resign / 23.0% flag -- a mate-only taxonomy would censor 74% of decisive games).
#
# TIME is the one ending that is NOT board-caused: a flagged position may be completely winning,
# so its terminal says nothing about the geometry. It is CENSORED -- given no pole, and excluded
# from every terminal term. Its positions still supply ordinary reachability pairs; only the
# ending is dropped. This is the same censoring rule censored_plies_loss encodes.
TERM_NONE, TERM_TIME = -1, -2
# EVERY LABEL HERE WAS CHECKED AGAINST THE DATA, and two first drafts were wrong:
#   * there is no WIN_MATE. Terminals are MOVER-POV (as everywhere else in the repo), and measured
#     over 259 mates the side to move loses in 1.000 of them -- a checkmate is never a win for the
#     player on move, so a WIN_MATE pole would have stayed empty forever.
#   * RESIGN had to SPLIT. Measured over 266 resignations the mover loses in only 0.883; in 0.117
#     the mover is the WINNER (you may resign on the opponent's turn). One RESIGN pole would have
#     put an eighth of them on the wrong side of the outcome level.
#   * agreed vs adjudicated draws are different objects: of 491 rule-free draws, 487 were SF games
#     (engine adjudication) and 4 human (actual agreement). Merging them would have labelled the
#     engine pool's dominant ending with a human behaviour.
WIN, DRAW, LOSS = 0, 1, 2
TERMINALS = ["LOSS_MATE", "RESIGN_LOSS", "RESIGN_WIN", "DRAW_AGREED", "DRAW_ADJUDICATED",
             "DRAW_FIFTY", "DRAW_STALEMATE", "DRAW_INSUFFICIENT", "DRAW_REPETITION"]
TERM_ID = {n: i for i, n in enumerate(TERMINALS)}
# RESIGN_* and DRAW_AGREED are judgments rather than rules, but all are caused by the POSITION
# (a player reads the board and concludes), which is exactly the property TIME lacks.
TERM_OUTCOME = np.array([LOSS, LOSS, WIN, DRAW, DRAW, DRAW, DRAW, DRAW, DRAW], np.int8)

# CERTAINTY RADIUS, in plies, of a terminal instance from its pole. This is the one place the
# hierarchy admits that not all terminals are equally trustworthy, and it is a PRIOR we are
# choosing, stated rather than buried.
#
#   0.0  A RULE FIRED. The position IS that outcome, by law -- mate, stalemate, insufficient
#        material, threefold, fifty-move. Radius 0 means exact subsumption: d(instance -> pole) = 0
#        while d(pole -> instance) stays large. Nothing weaker is true of these, and nothing
#        stronger is available.
#
#   0.0  SF-ADJUDICATED DRAWS (Kaveh 2026-08-05: "draw adjudicated should be agreed as a draw
#        position near a drawn pole if SF says so"). No rule fired, but full-strength deterministic
#        Stockfish on both sides called it drawn, and in this corpus that IS the ground truth we
#        have for drawnness. Trusting it is what makes the SF pool's 76% draw majority usable
#        evidence instead of 41% of terminals discarded as an artifact.
#
#   0.0  ALL DRAWS, including draw-by-agreement (Kaveh 2026-08-05: "all draws are truly near drawn
#        poles"). A game that ended drawn ended in the draw basin however it got there, so every
#        draw terminal -- rule, engine adjudication, or human agreement alike -- subsumes into the
#        draw pole. This is a claim about the OUTCOME basin, not about either player's judgment
#        being correct move by move.
#
#   1.0  RESIGNATION only. This is the one terminal where the position and the result can genuinely
#        disagree: measured, 11.7% of resignations happen on the WINNER's turn, so a resigned
#        position is not reliably the lost position its result says it is. It sits one ply OFF the
#        pole -- near enough to carry the outcome, far enough that the geometry is not asked to
#        treat a club player's opinion as a theorem. (pole_radial_anchor's existing target,
#        log1p(1), is exactly this shell.)
TERM_RADIUS = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32)

# Class weights for the terminal terms. SF adjudication is 41% of all terminals while stalemate is
# 0.5%, and an unweighted anchor loss would let the majority class set the whole pole geometry and
# drown the rare endings -- which are the informative ones for per-endgame competence. Inverse
# frequency, computed per store and capped, keeps the rare terminals visible without letting a
# six-instance class dominate. See Trajectories.terminal_weights().
TERM_WEIGHT_CAP = 8.0

_CACHE_KEYS = ("tok", "glob", "start", "length", "game_id", "source", "result", "elo",
               "term", "mat")


def classify_terminal(board, flagged: bool):
    """-> terminal type id, or TERM_TIME (censored) / TERM_NONE (game did not actually end).

    Read off the FINAL BOARD, not off the result label, because the label conflates a mate with a
    resignation with a flag -- and those are three different objects here. Order matters: mate and
    stalemate are checked before the claimable draws, since a position can satisfy both the
    fifty-move count and be mate, and the mate is what happened.
    """
    if board.is_checkmate():
        # mover-POV, matching the rest of the repo: the side to move has been mated, so the
        # position is a LOSS for whoever is on move.
        return TERM_ID["LOSS_MATE"]
    if board.is_stalemate():
        return TERM_ID["DRAW_STALEMATE"]
    if board.is_insufficient_material():
        return TERM_ID["DRAW_INSUFFICIENT"]
    if board.is_repetition(3):
        return TERM_ID["DRAW_REPETITION"]
    if board.halfmove_clock >= 100:
        return TERM_ID["DRAW_FIFTY"]
    if flagged:
        return TERM_TIME                                 # censored: not board-caused
    return TERM_NONE                                     # resignation / agreement -- see _replay_one


def _replay_one(args):
    """Replay one game to (tok (P,64), glob (P,6)) uint8 -- every ply, including the start position.

    Runs in a worker process, so chess/tokenize are imported inside (cloudpickle serialises
    referenced globals by value; see scaffold.py's note on the same hazard).
    """
    ucis, max_plies, tb_pieces, flagged, result, is_sf, with_planes = args
    import chess
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize
    if with_planes:
        # LczeroBoard subclasses chess.Board, so tokenize() is unaffected; it just also knows how
        # to emit the lc0 112-plane input. Planes are produced in THIS pass rather than a second
        # one so they cannot drift out of alignment with the token rows -- a second replay would
        # have to reproduce every drop/truncation decision exactly to stay row-aligned.
        from lczerolens import LczeroBoard
        b = LczeroBoard()
    else:
        b = chess.Board()
    planes = []
    toks, globs = [], []
    t, g = tokenize(b)
    toks.append(t); globs.append(g)                     # ply 0 = the initial position
    if with_planes:
        planes.append(b.to_input_tensor())
    truncated = len(ucis) > max_plies                    # the real ending is past our cut
    for u in ucis[:max_plies]:
        try:
            b.push(chess.Move.from_uci(u))
        except Exception:
            truncated = True
            break                                        # malformed move list: keep the prefix
        t, g = tokenize(b)
        toks.append(t); globs.append(g)
        if with_planes:
            planes.append(b.to_input_tensor())
        # OPTIONAL tablebase handoff, DEFAULT OFF (tb_pieces=0). Kaveh 2026-08-05 first asked to
        # stop at the 5-piece boundary ("we don't need training data for <=5 pieces") and then
        # reversed it -- "carry it to the end" -- so games run to their true end. The switch is
        # kept because the argument for it is real: below the boundary the answer is looked up
        # exactly rather than learned, so training there approximates a solved function. Set
        # tb_pieces=5 to truncate at the first position with <=5 pieces (that position is KEPT --
        # it is the handoff point and the last thing the learned field is responsible for).
        if tb_pieces and int((t > 0).sum()) <= tb_pieces:
            truncated = True
            break
    if len(toks) < 2:
        return None
    term = classify_terminal(b, bool(flagged))
    if term == TERM_NONE:
        # No rule fired and the clock did not run out, so the game was ended by judgment.
        if result != 0:
            # Resignation. Which SIDE resigned is not assumable -- measured, the mover is the loser
            # only 88.3% of the time -- so it is read off the result relative to the side to move.
            mover_loses = (result == -1) if b.turn else (result == 1)   # b.turn: True = white
            term = TERM_ID["RESIGN_LOSS" if mover_loses else "RESIGN_WIN"]
        else:
            term = TERM_ID["DRAW_ADJUDICATED" if is_sf else "DRAW_AGREED"]
    if truncated:
        term = TERM_NONE          # we stopped early; this is not the game's real terminal
    # Material signature of the FINAL position: counts of the 12 piece types. This is the
    # "endgame type" a specific pole stands for (KBNvK, KRvK, ...), taken from the tokens so no
    # chess library or plane decoding is involved.
    last = toks[-1]
    mat = np.array([(last == p).sum() for p in range(1, 13)], np.uint8)
    pk = None
    if with_planes:
        from catspace.research.components.encoder.approaches.reach_probability.src.lc0_prefix import (
            pack_planes)
        import torch
        pk = pack_planes(torch.stack(planes))
    return (np.stack(toks).astype(np.uint8), np.stack(globs).astype(np.uint8),
            np.int8(term), mat, pk)


def load_human_games(n, seed, records=None, max_plies=400):
    """-> list[(game_id, result, ucis, flagged)] sampled uniformly across ALL shards.

    `flagged` comes from the `termination` column and is the CENSORING flag: a Time-forfeit game
    ended for a reason the board does not explain, so it gets no terminal pole. It is 23.5% of this
    corpus (measured: 46,953 of 200,000 in shard 0), which is far too large a slice to leave
    silently mixed in with real board-caused endings.
    """
    import pyarrow.parquet as pq
    records = records or paths.records("lichess_2019-01")
    files = sorted(glob.glob(os.path.join(str(records), "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet under {records}")
    rng = np.random.default_rng(seed)
    per = max(1, n // len(files))
    out = []
    for f in files:
        m = pq.read_metadata(f).num_rows
        take = np.sort(rng.choice(m, size=min(per, m), replace=False))
        d = pq.read_table(f, columns=["game_id", "result", "moves", "termination", "white_elo", "black_elo"]).to_pydict()
        for r in take:
            out.append((int(d["game_id"][r]), int(d["result"][r]),
                        d["moves"][r].split()[:max_plies],
                        d["termination"][r] == "Time forfeit",
                        (int(d["white_elo"][r]), int(d["black_elo"][r]))))
        if len(out) >= n:
            break
    return out[:n]


def load_sf_games(n, seed, tsv=None, max_plies=400):
    """-> list[(game_id, result, ucis, flagged)]. Engine games never flag, so flagged is always
    False -- the SF pool contributes only board-caused terminals by construction."""
    tsv = tsv or paths.derived("opening_pool_sfsf_moves.tsv")
    lines = [l for l in open(tsv) if l.strip()]
    rng = np.random.default_rng(seed)
    if n < len(lines):
        lines = [lines[i] for i in np.sort(rng.choice(len(lines), size=n, replace=False))]
    out = []
    for ln in lines:
        p = ln.rstrip("\n").split("\t")
        if len(p) >= 3:
            out.append((int(p[0]), int(p[1]), p[2].split()[:max_plies], False, (SF_ELO, SF_ELO)))
    return out


def position_hash(tok: np.ndarray, glob: np.ndarray) -> np.ndarray:
    """(N,) uint64 FNV-1a hash over the 70 bytes that DEFINE the position.

    Position identity has to include castling rights and the en-passant file, not just the piece
    placement -- two boards with the same pieces but different ep rights are different positions and
    are not a repetition. That is exactly what a plane-derived key gets wrong (ep is encoded nowhere
    in the lc0 112 planes, verified), and it is why identity is taken from the TOKENS here.

    A 64-bit hash over ~2e7 positions has a birthday collision probability of ~1e-5, and a collision
    only matters if it lands inside one game; the cost of the rare false repetition is one spurious
    training pair, not a wrong label anywhere in the analysis.
    """
    h = np.full(len(tok), np.uint64(0xCBF29CE484222325), dtype=np.uint64)
    prime = np.uint64(0x100000001B3)
    for col in (tok, glob):
        for c in range(col.shape[1]):
            h = (h ^ col[:, c].astype(np.uint64)) * prime
    return h


@dataclass
class Trajectories:
    """Flat every-ply store. Position i of game g is row `start[g] + i`."""
    tok: np.ndarray          # (N,64) uint8   piece ids, vocab 0..12
    glob: np.ndarray         # (N,6)  uint8   [turn, K, Q, k, q, ep_file+1]
    start: np.ndarray        # (G,)   int64   first row of each game
    length: np.ndarray       # (G,)   int32   plies (rows) per game
    game_id: np.ndarray      # (G,)   int64
    source: np.ndarray       # (G,)   int8    HUMAN / SF
    result: np.ndarray       # (G,)   int8
    # (G,2) int16 [white_elo, black_elo]. Real ratings for human games, SF_ELO for engine games --
    # this is what makes the conditioning "this 1400 vs this 2200" instead of a human/engine flag.
    elo: np.ndarray
    term: np.ndarray         # (G,)   int8    TERMINALS index, or TERM_TIME / TERM_NONE
    mat: np.ndarray          # (G,12) uint8   piece counts of the FINAL position (endgame type)
    # (N,889) uint8 bit-packed lc0 112-plane inputs, or None when the token path is in use.
    # 889 B/position against 7168 raw (8.06x); see lc0_prefix.pack_planes for why exactly one
    # plane (rule50) needs a byte and the other 111 pack to bits.
    planes: np.ndarray | None = None

    def __len__(self):
        return len(self.start)

    @property
    def n_positions(self):
        return len(self.tok)

    def game_of_row(self):
        """(N,) int32 game index per row -- for split-by-game masks and cross-game checks."""
        return np.repeat(np.arange(len(self.start), dtype=np.int32), self.length)

    def ply_of_row(self):
        """(N,) int32 ply index within its game (0 = initial position)."""
        return (np.arange(self.n_positions, dtype=np.int64)
                - np.repeat(self.start, self.length)).astype(np.int32)

    def elo_of_row(self):
        """(N,) float32 Elo of the SIDE TO MOVE at each position.

        Side-to-move, not an average: the conditioning asks "how likely is THIS player to err from
        here", and the player on move is the one about to make the mistake. glob[:,0] is the turn
        flag (1 = white)."""
        per_game = np.repeat(self.elo, self.length, axis=0)          # (N,2)
        white_to_move = self.glob[:, 0].astype(bool)
        return np.where(white_to_move, per_game[:, 0], per_game[:, 1]).astype(np.float32)

    def outcome_of_row(self):
        """(N,) int8 MOVER-POV outcome per position: WIN / DRAW / LOSS, or -1 = CENSORED.

        This is the label the basin cross-entropy consumes, and it is the ONLY supervision the
        three-pole readout needs -- no negatives, no unreachability labels, just "which pole did
        the game this position came from actually end at". A position seen in games that were won
        60% of the time converges to reading 60%, because CE over a softmax is a proper scoring
        rule; the competition lives in the softmax denominator rather than in manufactured
        negative pairs.

        Mover-POV, matching the rest of the repo: the same board is a WIN for the side to move and
        a LOSS for the other, so the label has to be relative to whoever is on move or the two
        colours cancel and every position reads as a draw.

        TIME-FORFEIT games are CENSORED to -1 and take no part: a flagged position may be
        completely winning, so its recorded result says nothing about the board. Same rule the
        terminal poles use, applied to every ply rather than just the last one.
        """
        res = np.repeat(self.result, self.length).astype(np.int64)
        censored = np.repeat(self.term == TERM_TIME, self.length)
        white_to_move = self.glob[:, 0].astype(bool)
        mover_wins = np.where(white_to_move, res == 1, res == -1)
        mover_loses = np.where(white_to_move, res == -1, res == 1)
        out = np.full(self.n_positions, DRAW, np.int8)
        out[mover_wins] = WIN
        out[mover_loses] = LOSS
        out[censored] = -1
        return out

    def piece_count(self):
        """(N,) int16 total pieces. ANALYSIS-TIME LABEL ONLY -- never a model input. Straight from
        the tokens (non-empty squares), so no chess library and no plane decoding is involved."""
        return (self.tok > 0).sum(1).astype(np.int16)

    def terminal_rows(self):
        """(rows, term) for games with a BOARD-CAUSED ending -- the terminal instances.

        Time-forfeit games (TERM_TIME) and truncated ones (TERM_NONE) are dropped: a flagged
        position may be dead winning, so anchoring it to a draw/loss pole would teach the geometry
        something false. Their positions still take part in every reachability pair; only their
        ending is censored.
        """
        ok = np.flatnonzero(self.term >= 0)
        return (self.start[ok] + self.length[ok] - 1).astype(np.int64), self.term[ok]

    def terminal_weights(self):
        """(len(TERMINALS),) float32 class weights, inverse-frequency, capped and mean-normalised.

        SF adjudication is ~41% of terminals and stalemate ~0.5%; unweighted, the majority class
        would set the entire pole geometry and the rare endings would contribute nothing. Those
        rare ones are exactly the informative ones for per-endgame competence ("sf won't blunder a
        stalemate but a human might"), so they have to survive the aggregate. Capped at
        TERM_WEIGHT_CAP so a six-instance class cannot dominate in the other direction.
        """
        cnt = np.bincount(self.term[self.term >= 0], minlength=len(TERMINALS)).astype(np.float64)
        w = np.where(cnt > 0, cnt.sum() / np.maximum(cnt, 1) / max(len(TERMINALS), 1), 0.0)
        w = np.minimum(w, TERM_WEIGHT_CAP)
        m = w[w > 0].mean() if (w > 0).any() else 1.0
        return (w / m).astype(np.float32)

    def radial_targets(self):
        """(n_terminal_instances,) log-space target radius per terminal, for pole_radial_anchor.

        log1p(TERM_RADIUS[term]): 0 for every rule-fired ending, every draw (all draws are near the
        draw pole), and SF-adjudicated draws; log1p(1) for resignations, the one terminal where
        position and result measurably disagree.
        """
        _, term = self.terminal_rows()
        return np.log1p(TERM_RADIUS[term]).astype(np.float32)

    def source_of_terminal(self):
        """(n_terminal_instances,) int64 dynamics id (HUMAN/SF) -- the pole conditioning input."""
        ok = np.flatnonzero(self.term >= 0)
        return self.source[ok].astype(np.int64)

    def ending_poles(self, exclude_material=False):
        """-> (rows, pole, names, parent) keyed on ENDING TYPE ONLY -- no material signature.

        Kaveh 2026-08-05: "if we do poles all the way up to the end, without stopping at 5 piece
        endgames, then there won't be material signature. And ending type is fine."

        He is right, and my earlier objection was overstated. The leak I worried about was
        (ending x MATERIAL SIGNATURE) poles, which key terminal identity on piece counts and would
        feed material straight into a run whose whole claim is that piece count is an analysis
        label and never an input. Ending type is a different object: "this game ended in stalemate"
        says nothing about any position's piece count. And because games run to their TRUE end
        (tb_pieces=0), every game has a well-defined ending type, so the signature was never needed
        for granularity in the first place.

        `exclude_material` DEFAULTS OFF, and the first draft had it on for a bad reason. It dropped
        DRAW_INSUFFICIENT because that category is NAMED after material -- a naming criterion, not
        a leakage one. Kaveh pushed back and the exclusion does not survive:

          * The pole tells the model no piece count. It says "these terminal positions belong
            together". If the model then works out WHY -- few pieces -- that is an INFERENCE, which
            is the thing being measured, not a leak.
          * It was inconsistent. DRAW_FIFTY is about the absence of captures and pawn moves;
            repetitions and mates both correlate with material configuration. If
            correlation-with-material disqualified a pole, nearly every pole would go.
          * Decisively: NO POLE ENCODES DIRECTION. The strata claim is that material never RISES --
            an ordering. Every pole is an equivalence class over terminal positions and contains no
            ordering whatever. The paired ratchet asks whether a source that COULD reach a target
            outscores one that could not, and grouping low-material draws supplies nothing about
            that.

        The flag is kept so the ablation is one argument away, but the default is to use every
        board-caused ending.

        Two levels: ending type >= outcome. Poles are the roots WIN/DRAW/LOSS plus one per surviving
        ending type, each subsuming into its outcome.
        """
        rows, term = self.terminal_rows()
        drop = {TERM_ID["DRAW_INSUFFICIENT"]} if exclude_material else set()
        keep = ~np.isin(term, list(drop)) if drop else np.ones(len(term), bool)
        rows, term = rows[keep], term[keep]
        names = ["WIN", "DRAW", "LOSS"]
        parent = [-1, -1, -1]
        slot = {}
        for t, nm in enumerate(TERMINALS):
            if t in drop:
                continue
            slot[t] = len(names)
            names.append(nm)
            parent.append(int(TERM_OUTCOME[t]))
        pole = np.array([slot[int(t)] for t in term], np.int64)
        return rows, pole, names, np.asarray(parent, np.int64)

    def endgame_poles(self, min_count=50):
        """-> (pole_of_game, names, parent) for the SUBSUMPTION hierarchy.

        Kaveh 2026-08-05: "we defined each endgame as a pole, each instance of that endgame as a
        point with 1-ply step to that pole" -- extended here to "all the win and mate types, except
        for time (flagging)", and to 0-ply (subsumption) rather than 1-ply, because an instance IS
        its endgame rather than being one move from it.

        Three levels, each dominated by the one above it in the IQE coordinates:

            terminal instance  >=  (ending x material signature)  >=  ending type  >=  outcome

        A specific (ending, signature) pole is only created when at least `min_count` games end
        there; rarer signatures fall back to their ending-type pole. That is the same thin-bucket
        fallback the conformal Mondrian code uses, and it is reported rather than hidden -- a pole
        supported by six games would be fitting noise and would make the per-endgame competence
        readout meaningless exactly where it looks most interesting.
        """
        rows, term = self.terminal_rows()
        ok = np.flatnonzero(self.term >= 0)
        sig = self.mat[ok]                                # (n,12) piece counts of the final position
        key = np.concatenate([term[:, None].astype(np.int64), sig.astype(np.int64)], 1)
        uniq, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        big = cnt[inv] >= min_count
        names, parent, pole = [], [], np.full(len(ok), -1, np.int64)
        # level 3 (outcome) then level 2 (ending type) then level 1 (specific endgame)
        for o, nm in enumerate(("WIN", "DRAW", "LOSS")):
            names.append(nm); parent.append(-1)
        base_t = len(names)
        for t, nm in enumerate(TERMINALS):
            names.append(nm); parent.append(int(TERM_OUTCOME[t]))
        for u in np.flatnonzero(np.bincount(inv, minlength=len(uniq)) >= min_count):
            t = int(uniq[u][0])
            names.append(f"{TERMINALS[t]}|{''.join(map(str, uniq[u][1:]))}")
            parent.append(base_t + t)
        spec = {int(u): base_t + len(TERMINALS) + i
                for i, u in enumerate(np.flatnonzero(np.bincount(inv, minlength=len(uniq)) >= min_count))}
        for i in range(len(ok)):
            pole[i] = spec.get(int(inv[i]), base_t + int(term[i])) if big[i] \
                else base_t + int(term[i])
        return rows, pole, names, np.asarray(parent, np.int64)

    def repeats(self):
        """(r1, r2) int64 row pairs, r1 < r2, SAME game, IDENTICAL position -- the repetitions.

        Every consecutive occurrence is emitted (a threefold gives (1st,2nd) and (2nd,3rd), whose
        transitive closure covers (1st,3rd) through the coverage cummax below), so the cost is
        linear in occurrences rather than quadratic in the multiplicity.
        """
        h = position_hash(self.tok, self.glob)
        g = self.game_of_row().astype(np.int64)
        rows = np.arange(self.n_positions, dtype=np.int64)
        order = np.lexsort((rows, h, g))                  # group by (game, position), ply-ascending
        gs, hs, rs = g[order], h[order], order
        same = (gs[1:] == gs[:-1]) & (hs[1:] == hs[:-1])
        r1, r2 = rs[:-1][same], rs[1:][same]
        return r1, r2

    def coverage(self):
        """(N,) int64 `maxr2_from[i]` = the furthest row reachable BACK to i via a repetition.

        A backward pair (x_j -> x_i), i < j, is genuinely OBSERVED-reachable iff some repetition
        (r1, r2) satisfies r1 <= i and j <= r2: play forward from x_j to x_r2, which is the same
        position as x_r1, then replay the game's own moves r1 -> i. So the test is `j <= cov[i]`,
        and cov is a per-game running max of r2 seeded at each r1.

        The running max is taken GLOBALLY rather than per game, which is safe because a game's
        values never exceed its own last row and the next game's rows all start above it -- so a
        leaked value can never satisfy `j <= cov[i]` for a later game.
        """
        r1, r2 = self.repeats()
        cov = np.full(self.n_positions, -1, np.int64)
        if len(r1):
            np.maximum.at(cov, r1, r2)
        return np.maximum.accumulate(cov)


def build(n_human=100_000, n_sf=100_000, seed=0, workers=None, cache=True,
          records=None, tsv=None, max_plies=400, tb_pieces=0, with_planes=False,
          verbose=True) -> Trajectories:
    """Replay and tokenize every ply of `n_human` + `n_sf` games. Cached on disk by (n, seed).

    `tb_pieces` (default 0 = off) truncates each game at the first position with <= that many
    pieces. Games run to their true END by default: the deep endgame is where a population's
    competence differs most sharply by ENDGAME TYPE (Kaveh 2026-08-05: "people might bungle one
    endgame type over another, e.g. knight bishop king"), which is the signal the endgame-pole
    readout exists to measure. Throwing it away to save a solved-by-tablebase region would discard
    the most legible weakness data in the corpus.
    """
    # v2 in the key: the schema gained per-game Elo, so a v1 cache would load without it.
    key = hashlib.blake2b(f"v2|{n_human}|{n_sf}|{seed}|{max_plies}|{tb_pieces}|{int(with_planes)}"
                          .encode(), digest_size=8).hexdigest()
    cdir = paths.derived(f"cache/traj_{key}")
    if cache and os.path.exists(os.path.join(cdir, "tok.npy")):
        if verbose:
            print(f"[traj] cache hit {cdir}", flush=True)
        d = {k: np.load(os.path.join(cdir, f"{k}.npy"))
             for k in _CACHE_KEYS}
        return Trajectories(**d)

    t0 = time.time()
    games = ([(g, HUMAN) for g in load_human_games(n_human, seed, records, max_plies)]
             + [(g, SF) for g in load_sf_games(n_sf, seed, tsv, max_plies)])
    if verbose:
        print(f"[traj] {len(games):,} games loaded [{time.time()-t0:.0f}s]; replaying every ply...",
              flush=True)

    workers = workers or max(1, (os.cpu_count() or 4) - 2)
    payload = [(ucis, max_plies, tb_pieces, flg, res, src == SF, with_planes)
               for (_, res, ucis, flg, _el), src in games]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        done = list(ex.map(_replay_one, payload, chunksize=64))

    toks, globs, starts, lens, gids, srcs, ress, terms, mats = [], [], [], [], [], [], [], [], []
    pks, elos = [], []
    n = 0
    for ((gid, res, _, _, el), src), r in zip(games, done):
        if r is None:
            continue
        tk, gb, term, mat, pk = r
        toks.append(tk); globs.append(gb)
        if pk is not None:
            pks.append(pk)
        starts.append(n); lens.append(len(tk)); n += len(tk)
        gids.append(gid); srcs.append(src); ress.append(res); elos.append(el)
        terms.append(term); mats.append(mat)

    tr = Trajectories(
        tok=np.concatenate(toks), glob=np.concatenate(globs),
        start=np.asarray(starts, np.int64), length=np.asarray(lens, np.int32),
        game_id=np.asarray(gids, np.int64), source=np.asarray(srcs, np.int8),
        result=np.asarray(ress, np.int8), elo=np.asarray(elos, np.int16),
        term=np.asarray(terms, np.int8),
        mat=np.stack(mats).astype(np.uint8),
        planes=np.concatenate(pks) if pks else None)
    if verbose:
        cens = int((tr.term == TERM_TIME).sum())
        named = {TERMINALS[i]: int((tr.term == i).sum()) for i in range(len(TERMINALS))}
        print(f"[traj] {len(tr):,} games | {tr.n_positions:,} positions "
              f"({tr.tok.nbytes/2**30:.2f} GB tokens) [{time.time()-t0:.0f}s]", flush=True)
        print(f"[traj] terminals {named} | CENSORED(time) {cens:,} "
              f"({cens/max(len(tr),1):.1%}) | none {int((tr.term==TERM_NONE).sum()):,}", flush=True)
    if cache:
        os.makedirs(cdir, exist_ok=True)
        for k in _CACHE_KEYS:
            np.save(os.path.join(cdir, f"{k}.npy"), getattr(tr, k))
        if tr.planes is not None:
            np.save(os.path.join(cdir, "planes.npy"), tr.planes)
        if verbose:
            print(f"[traj] cached -> {cdir}", flush=True)
    return tr


class PairSampler:
    """Fresh pairs every step from the games in `games`, with no pair dataset materialised.

    Every ply gives ~2,300 ordered pairs per game against 10 in the old 5-ply sample, so the pair
    set is effectively unbounded: freezing a subset of it into a file would throw away most of the
    signal AND fix the sampling noise, which is the opposite of what resampling per step gives.

    `games` is a game-index array, so the train/cal/test split is enforced HERE, at the only place
    pairs come from. A sampler built on the training games can never emit a calibration pair.
    """

    def __init__(self, tr: Trajectories, games: np.ndarray, seed: int = 0,
                 cov: np.ndarray | None = None, repeats: tuple | None = None,
                 min_ply: int = 0):
        """`min_ply` DROPS the first plies of every game, and for the dynamics-conditioned
        readouts it is not optional -- it must be 8.

        THE SF POOL IS NOT SF BEFORE PLY 8 (Kaveh 2026-08-05: "the sf-sf games start at ply 8").
        gen_field_positions_v2.py records the provenance: the SF-vs-SF games are `prefix + SF
        continuation`, where the prefix is drawn from the top-100k most FREQUENT human ply-8
        openings. So plies 0..7 of every SF game are HUMAN moves, and below ply 8 the two
        populations differ in opening COMPOSITION (head of the human distribution vs all of it)
        rather than in dynamics.

        Two things break if this is ignored, and neither shows up as a bad loss:
          * d_mistake becomes incoherent. Its whole definition is "extra distance relative to BEST
            play", and the base would be fitted partly on human opening moves labelled SF -- so the
            residual would measure human error against a partly-human reference and understate it.
          * any human-vs-SF pole divergence below ply 8 is an opening-composition artifact. It
            would read as "the dynamics differ here", which is exactly the finding we would want to
            claim, arriving for free from a sampling detail.

        The STRATA question is unaffected and correctly uses min_ply=0: it pools both populations,
        never reads the source label, and compares within-game ply-matched reversals, so opening
        composition cannot manufacture a material-ratchet differential.
        """
        self.tr = tr
        self.rng = np.random.default_rng(seed)
        self.min_ply = int(min_ply)
        games = np.asarray(games, np.int64)
        if self.min_ply:                      # a game must still have room for a triple after the cut
            games = games[tr.length[games] > self.min_ply + 2]
        self.games = games
        self.start = tr.start[self.games] + self.min_ply
        self.length = tr.length[self.games].astype(np.int64) - self.min_ply
        w = np.maximum(self.length - 1, 0).astype(np.float64)   # a 1-ply game has no forward pair
        self.cum = np.cumsum(w)
        self.total = float(self.cum[-1]) if len(self.cum) else 0.0
        w3 = np.maximum(self.length - 2, 0).astype(np.float64)  # and no triple below 3 plies
        self.cum3 = np.cumsum(w3)
        self.total3 = float(self.cum3[-1]) if len(self.cum3) else 0.0
        self.cov = tr.coverage() if cov is None else cov
        r1, r2 = tr.repeats() if repeats is None else repeats
        keep = np.isin(tr.game_of_row()[r1], self.games)        # repetitions inside these games only
        self.r1, self.r2 = r1[keep], r2[keep]
        # rows of the allowed games, flat -- the cross-game sampler draws from these. Built by
        # arithmetic rather than a per-game arange loop, which at 200k games is not free.
        off = np.cumsum(self.length) - self.length
        self.rows = (np.arange(int(self.length.sum()), dtype=np.int64)
                     - np.repeat(off, self.length) + np.repeat(self.start, self.length))
        self.game_of_row = tr.game_of_row()

    def _pick_games(self, n):
        u = self.rng.random(n) * self.total
        return np.searchsorted(self.cum, u, side="right").clip(0, len(self.games) - 1)

    def forward(self, n):
        """(ia, ib, gap): a before b in the same game. OBSERVED reachable, gap plies apart."""
        k = self._pick_games(n)
        L = self.length[k]
        i = (self.rng.random(n) * (L - 1)).astype(np.int64)
        j = i + 1 + (self.rng.random(n) * (L - 1 - i)).astype(np.int64)
        s = self.start[k]
        return s + i, s + j, (j - i).astype(np.int64)

    def triples(self, n):
        """(i, j, k) rows, ply i < j < k in ONE game -- the unit the objective is written on.

        A triple, rather than a pair, because the repulsion terms are PAIRED: they ask whether an
        unobserved target sits further from the source than an observed one *of the same source*.
        Sampling a source with both a real future (k) and a real past (i) supplies exactly that,
        and it does so from three encoder rows instead of four.
        """
        u = self.rng.random(n) * self.total3
        g = np.searchsorted(self.cum3, u, side="right").clip(0, len(self.games) - 1)
        L = self.length[g]
        i = (self.rng.random(n) * (L - 2)).astype(np.int64)
        j = i + 1 + (self.rng.random(n) * (L - 2 - i)).astype(np.int64)
        k = j + 1 + (self.rng.random(n) * (L - 1 - j)).astype(np.int64)
        s = self.start[g]
        return s + i, s + j, s + k

    def uncovered(self, ia, ib):
        """(mask,) True where the reversal (x_ib -> x_ia) is NOT observed via any repetition.

        ia < ib expected. See Trajectories.coverage() for why `ib <= cov[ia]` is the exact test.
        """
        return ib > self.cov[ia]

    def reversible(self, n):
        """(ia, ib, gap): OBSERVED backward pairs, via repetitions. Empty if the games have none.

        From a repetition (r1, r2), any row k in (r1, r2] reaches x_r1 in r2 - k plies, because
        x_r2 IS x_r1. This is the data-grounded evidence of reversibility -- nothing here assumes
        that any position can be returned to.
        """
        if not len(self.r1):
            return (np.zeros(0, np.int64),) * 3
        t = self.rng.integers(0, len(self.r1), n)
        r1, r2 = self.r1[t], self.r2[t]
        k = r1 + 1 + (self.rng.random(n) * (r2 - r1)).astype(np.int64)
        return k, r1, (r2 - k).astype(np.int64)

    def backward(self, n):
        """(ia, ib): (x_j -> x_i), i < j, with NO repetition covering [i, j]. Never observed.

        This is the ratchet's raw material, and it is deliberately NOT called a negative: the game
        did not walk back, which is not the same as it being impossible. It carries a repulsion.
        """
        ia, ib, _ = self.forward(n)
        ok = ib > self.cov[ia]                                # not covered by a repetition
        return ib[ok], ia[ok]                                 # reversed: source = the later row

    def cross(self, ia, max_redraw=8):
        """(ib,) one partner row per source in `ia`, from a DIFFERENT game. Never observed.

        Returns exactly len(ia) partners rather than filtering, because the repulsion terms are
        paired PER SOURCE -- dropping the collisions would silently misalign the far target with
        the source whose region it is supposed to sit outside of. Same-game draws are redrawn; at
        1/|games| they are rare, and after `max_redraw` the residue is nudged to a neighbouring row
        rather than left wrong. Same-game draws are redrawn; a single draw collides with
        probability ~1/|games|, so after `max_redraw` rounds the residue is ~|games|^-8 -- reported
        by the caller's `cross_same_game` gate rather than assumed away.
        """
        ia = np.asarray(ia, np.int64)
        ib = self.rows[self.rng.integers(0, len(self.rows), len(ia))]
        for _ in range(max_redraw):
            bad = self.game_of_row[ia] == self.game_of_row[ib]
            if not bad.any():
                break
            ib[bad] = self.rows[self.rng.integers(0, len(self.rows), int(bad.sum()))]
        return ib
